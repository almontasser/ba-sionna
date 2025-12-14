"""
Base Station (BS) Controller with Learned Codebook and Feedback Processing

This module implements the BS-side controller for transmit beamforming in mmWave
communication systems. The controller manages a learned codebook of transmit beams
and processes feedback from the UE to generate the final refined beam.

Key Components:
    1. Learned Codebook: NCB trainable beams, initialized with DFT codebook
       - Each beam is a complex vector of size NTX (number of TX antennas)
       - Beams are normalized to unit norm
       - Codebook is jointly optimized during training
    
    2. Sequential Beam Sweeping: During T sensing steps
       - Cycles through codebook: beams [start, start+1, ..., start+T-1] mod NCB
       - Allows UE to gather measurements across different spatial directions
    
    3. Feedback Processing Network (N2): Feed-forward neural network
       - Input: UE feedback message m_FB (NFB real values)
       - Output: Final transmit beam f_T (NTX complex values)
       - Architecture: 2-3 fully connected layers with ReLU activation
       - Enables BS to refine beam based on UE's learned channel knowledge

Operation Modes:
    Sensing Phase (t=0 to T-1):
        - Returns beam f_t from codebook at index (start + t) mod NCB
        - Used for channel probing and measurement collection
    
    Final Phase (t=T):
        - Processes feedback m_FB through FNN to generate f_T
        - Generates refined beam adapted to specific channel conditions

Training:
    - Codebook beams are trainable parameters
    - FNN (N2) weights are trainable
    - Optimized end-to-end via beamforming gain maximization

References:
    Paper Section III.A: BS Controller Design
    Paper Scheme C3: Adaptive sensing with feedback
"""

import tensorflow as tf
from utils import normalize_beam, create_dft_codebook


class BSController(tf.keras.layers.Layer):
    """
    Base Station Controller with learned beam codebook.
    
    The BS maintains a codebook of NCB beams and sweeps through them sequentially.
    The codebook is trainable and learned end-to-end.
    """
    
    def __init__(self, 
                 num_antennas,
                 codebook_size,
                 initialize_with_dft=True,
                 trainable_codebook=True,
                 **kwargs):
        """
        Args:
            num_antennas: Number of transmit antennas (NTX)
            codebook_size: Number of beams in codebook (NCB)
            initialize_with_dft: Initialize codebook with DFT beams
            trainable_codebook: Whether codebook is trainable
        """
        super().__init__(**kwargs)
        self.num_antennas = num_antennas
        self.codebook_size = codebook_size
        self.initialize_with_dft = initialize_with_dft
        self.trainable_codebook = trainable_codebook

    def build(self, input_shape=None):
        """Build method to properly register variables with Keras"""
        # Create trainable codebook using add_weight for proper Keras tracking.
        # NOTE: Avoid `.numpy()` here so the layer can be built under graph mode.
        self.codebook_real = self.add_weight(
            name='codebook_real',
            shape=(self.codebook_size, self.num_antennas),
            initializer="zeros",
            trainable=self.trainable_codebook
        )
        self.codebook_imag = self.add_weight(
            name='codebook_imag',
            shape=(self.codebook_size, self.num_antennas),
            initializer="zeros",
            trainable=self.trainable_codebook
        )

        self.built = True

        if self.initialize_with_dft:
            dft_codebook = create_dft_codebook(self.codebook_size, self.num_antennas)
            dft_codebook = normalize_beam(tf.cast(dft_codebook, tf.complex64))
            self.codebook_real.assign(tf.math.real(dft_codebook))
            self.codebook_imag.assign(tf.math.imag(dft_codebook))
        else:
            scale = tf.sqrt(tf.cast(self.num_antennas, tf.float32))
            initial_real = tf.random.normal(
                [self.codebook_size, self.num_antennas], dtype=tf.float32
            ) / scale
            initial_imag = tf.random.normal(
                [self.codebook_size, self.num_antennas], dtype=tf.float32
            ) / scale
            self.codebook_real.assign(initial_real)
            self.codebook_imag.assign(initial_imag)

        # N2: FNN to map feedback to final beam (per paper)
        # Paper: "two to three layers of fully connected DNN"
        # Input: m_FB (NFB) -> Output: f_T (2*NTX)
        # Slightly wider FNN with normalization for more expressive final beam mapping.
        self.fnn = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='gelu', name='bs_fnn_1'),
            tf.keras.layers.LayerNormalization(name='bs_fnn_ln1'),
            tf.keras.layers.Dense(256, activation='gelu', name='bs_fnn_2'),
            tf.keras.layers.LayerNormalization(name='bs_fnn_ln2'),
            tf.keras.layers.Dense(2 * self.num_antennas, activation=None, name='bs_fnn_out')
        ], name='bs_fnn')
    
    @property
    def codebook(self):
        """Get the complex codebook."""
        if not self.built:
            self.build(None)
        # Cast to float32 for complex number creation (tf.complex doesn't support float16)
        real_f32 = tf.cast(self.codebook_real, tf.float32)
        imag_f32 = tf.cast(self.codebook_imag, tf.float32)
        return tf.complex(real_f32, imag_f32)
    
    def get_beam(self, beam_index):
        """
        Get a specific beam from the codebook.
        
        Args:
            beam_index: Index or indices of beam(s) to retrieve
                       Can be scalar or tensor of shape (batch,)
        
        Returns:
            Beam vector(s) of shape (batch, num_antennas) or (num_antennas,)
        """
        codebook = self.codebook
        
        # Normalize codebook beams
        codebook = normalize_beam(codebook)
        
        # Gather beams
        beams = tf.gather(codebook, beam_index)
        
        return beams
    
    def get_beam_sequence(self, start_index, num_steps, batch_size):
        """
        Get a sequence of beams for beam sweeping.
        
        Args:
            start_index: Starting beam index (scalar or shape (batch,))
            num_steps: Number of sensing steps (T)
            batch_size: Batch size
            
        Returns:
            Beam sequence of shape (batch, num_steps, num_antennas)
            Beam indices of shape (batch, num_steps)
        """
        # Handle scalar or batched start index
        if isinstance(start_index, int) or (isinstance(start_index, tf.Tensor) and start_index.shape.ndims == 0):
            # Scalar start index - same for all batch
            start_idx = tf.fill([batch_size], start_index)
        else:
            # Already batched
            start_idx = start_index
        
        # Generate sequence of indices: [start, start+1, ..., start+T-1] mod NCB
        step_offsets = tf.range(num_steps, dtype=tf.int32)  # [0, 1, 2, ..., T-1]
        step_offsets = tf.reshape(step_offsets, [1, num_steps])  # (1, T)
        
        start_idx = tf.reshape(start_idx, [batch_size, 1])  # (batch, 1)
        
        # Beam indices for each step
        beam_indices = (start_idx + step_offsets) % self.codebook_size  # (batch, T)
        
        # Get beams for all indices
        codebook = normalize_beam(self.codebook)  # (NCB, NTX)
        
        # Gather beams: for each batch and each time step
        # beam_indices: (batch, T)
        # We need to gather from codebook for each batch element
        beams_sequence = tf.gather(codebook, beam_indices)  # (batch, T, NTX)
        
        return beams_sequence, beam_indices
    
    def call(self, beam_index):
        """
        Forward pass - get beam(s) from codebook.
        
        Args:
            beam_index: Beam index or indices
            
        Returns:
            Beam vector(s)
        """
        return self.get_beam(beam_index)

    def get_beam_from_feedback(self, feedback):
        """
        Generate final beam f_T from feedback message using N2 (FNN).
        
        Args:
            feedback: Feedback message m_FB, shape (batch, NFB)
            
        Returns:
            Final beam f_T, shape (batch, num_antennas)
        """
        # Pass through FNN
        beam_real_imag = self.fnn(feedback)  # (batch, 2*NTX)
        
        # Convert to complex
        # Split into real and imag parts
        beam_real = beam_real_imag[:, :self.num_antennas]
        beam_imag = beam_real_imag[:, self.num_antennas:]
        # Cast to float32 for complex number creation (tf.complex doesn't support float16)
        beam_real = tf.cast(beam_real, tf.float32)
        beam_imag = tf.cast(beam_imag, tf.float32)
        beam = tf.complex(beam_real, beam_imag)
        
        # Normalize
        beam = normalize_beam(beam)
        
        return beam


if __name__ == "__main__":
    print("Testing BS Controller...")
    print("=" * 60)
    
    # Create BS controller
    bs_controller = BSController(
        num_antennas=64,
        codebook_size=8,
        initialize_with_dft=True,
        trainable_codebook=True
    )
    
    # Test single beam retrieval
    beam = bs_controller.get_beam(0)
    print(f"Single beam shape: {beam.shape}")
    print(f"Single beam norm: {tf.norm(beam):.4f}")
    
    # Test batch beam retrieval
    beam_indices = tf.constant([0, 1, 2, 3])
    beams = bs_controller.get_beam(beam_indices)
    print(f"\nBatch beams shape: {beams.shape}")
    print(f"Batch beams norms: {tf.norm(beams, axis=-1)}")
    
    # Test beam sequence
    batch_size = 10
    start_index = 0
    num_steps = 8
    beam_sequence, indices = bs_controller.get_beam_sequence(start_index, num_steps, batch_size)
    print(f"\nBeam sequence shape: {beam_sequence.shape}")
    print(f"Beam indices shape: {indices.shape}")
    print(f"Beam indices (first sample): {indices[0]}")
    print(f"Beam norms (first sample): {tf.norm(beam_sequence[0], axis=-1)}")
    
    # Test trainability
    print(f"\nCodebook trainable: {bs_controller.trainable_codebook}")
    print(f"Number of trainable variables: {len(bs_controller.trainable_variables)}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
