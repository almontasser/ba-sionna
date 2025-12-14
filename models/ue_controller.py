"""
User Equipment (UE) Controller with Adaptive RNN-Based Beam Selection

This module implements the UE-side controller for adaptive receive beamforming in
mmWave communication systems. The controller uses a recurrent neural network (RNN)
to process sequential sensing measurements and generate optimal receive combining
vectors at each step.

Key Features:
    - Configurable multi-layer GRU/LSTM for temporal processing
    - Adaptive receive beam generation based on sensing history
    - Feedback message generation for BS final beam refinement
    - Learns to extract channel information from noisy measurements

Operation Flow:
    At each sensing step t:
    1. Input: received signal y_t (complex scalar), BS beam index x_t
    2. RNN processes a feature vector derived from [Re(y_t), Im(y_t), x_t]
       (x_t can be scalar-normalized or one-hot; time feature optional)
       → hidden state h_t
    3. Output layer generates: receive beam w_t (complex vector, NRX-dim)
    4. Feedback layer generates: feedback m_t (real vector, NFB-dim)
    5. Beam is normalized: w_t ← w_t / ||w_t||

Network Architecture:
    Input: features derived from [Re(y_t), Im(y_t), x_t] plus optional additions
           (e.g., one-hot beam index, time-step feature), controlled via Config.
    ↓
    N-layer GRU/LSTM (hidden_size per layer)
    ↓
    ├─→ Dense(2*NRX) → Beam Output (w_t)
    └─→ Dense(NFB) → Feedback Output (m_t)

Trainable via backpropagation through the entire sensing sequence,
optimizing the final beamforming gain.

References:
    Paper Section III.B: UE Controller Design
    "Two layers of gated recurrent units (GRU)"
"""

import tensorflow as tf
from utils import normalize_beam, real_to_complex_vector


class UEController(tf.keras.Model):
    """
    UE Controller with RNN for adaptive beam selection.
    
    At each sensing step t, the UE:
    1. Receives the signal y_t and beam index x_t from BS
    2. Updates its internal state using RNN
    3. Generates the receive combining vector w_t
    4. Optionally generates feedback for BS
    """
    
    def __init__(self,
                 num_antennas,
                 rnn_hidden_size=128,
                 rnn_type="GRU",
                 num_layers=2,
                 num_feedback=4,
                 codebook_size=8,
                 beam_index_encoding="scalar",
                 include_time_feature=False,
                 input_layer_norm=False,
                 output_layer_norm=False,
                 dropout_rate=0.0,
                 rnn_dropout=0.0,
                 rnn_recurrent_dropout=0.0,
                 **kwargs):
        """
        Args:
            num_antennas: Number of receive antennas (NRX)
            rnn_hidden_size: Hidden state size for RNN
            rnn_type: Type of RNN ("GRU" or "LSTM")
            num_feedback: Number of real-valued feedback values (NFB)
            codebook_size: BS codebook size for beam index normalization
        """
        super().__init__(**kwargs)
        self.num_antennas = num_antennas
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_type = rnn_type
        self.num_layers = int(num_layers)
        self.num_feedback = num_feedback
        self.codebook_size = codebook_size
        self.beam_index_encoding = str(beam_index_encoding).lower()
        self.include_time_feature = bool(include_time_feature)
        self.input_layer_norm = bool(input_layer_norm)
        self.output_layer_norm = bool(output_layer_norm)
        self.dropout_rate = float(dropout_rate)
        self.rnn_dropout = float(rnn_dropout)
        self.rnn_recurrent_dropout = float(rnn_recurrent_dropout)

        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.beam_index_encoding not in {"scalar", "one_hot"}:
            raise ValueError(
                "beam_index_encoding must be 'scalar' or 'one_hot', "
                f"got '{self.beam_index_encoding}'."
            )
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be >= 2 for beam index features.")
        
        if self.input_layer_norm:
            self.input_ln = tf.keras.layers.LayerNormalization(name="ue_input_ln")
        else:
            self.input_ln = None
        if self.output_layer_norm:
            self.output_ln = tf.keras.layers.LayerNormalization(name="ue_output_ln")
        else:
            self.output_ln = None
        if self.dropout_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name="ue_dropout")
        else:
            self.dropout = None

        # Recurrent core (paper default: 2 layers).
        #
        # Important implementation detail:
        # We keep the recurrent layers as an explicit list of cells and run them
        # manually in `process_step()`. This avoids Keras' internal
        # StackedRNNCells, which (depending on TF/Keras version) may not be
        # fully serializable with `tf.train.Checkpoint(model=...)` and can lead
        # to checkpoints that *omit* UE RNN weights.
        if rnn_type == "GRU":
            cells = []
            for i in range(self.num_layers):
                cell = tf.keras.layers.GRUCell(
                    rnn_hidden_size,
                    dropout=self.rnn_dropout,
                    recurrent_dropout=self.rnn_recurrent_dropout,
                    name=f"ue_gru_cell_layer{i+1}",
                )
                # Assign each cell as its own attribute so TF/Keras checkpointing
                # reliably tracks it (lists of layers are not always tracked).
                setattr(self, f"ue_gru_cell_layer{i+1}", cell)
                cells.append(cell)
            self.rnn_cells = cells
        elif rnn_type == "LSTM":
            cells = []
            for i in range(self.num_layers):
                cell = tf.keras.layers.LSTMCell(
                    rnn_hidden_size,
                    dropout=self.rnn_dropout,
                    recurrent_dropout=self.rnn_recurrent_dropout,
                    name=f"ue_lstm_cell_layer{i+1}",
                )
                setattr(self, f"ue_lstm_cell_layer{i+1}", cell)
                cells.append(cell)
            self.rnn_cells = cells
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layers
        # Beam generation: map hidden state to complex beam
        # Output 2*NRX values (real and imaginary parts)
        self.beam_output = tf.keras.layers.Dense(
            2 * num_antennas,
            activation=None,
            name='beam_output'
        )
        
        # Feedback generation (C3): output real feedback vector (NFB values)
        self.feedback_output = tf.keras.layers.Dense(
            num_feedback,
            activation=None,
            name='feedback_output'
        )
    
    def get_initial_state(self, batch_size):
        """
        Get initial hidden state for RNN.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state(s) for RNN (list of states for 2-layer RNN)
        """
        if self.rnn_type == "GRU":
            return [
                tf.zeros([batch_size, self.rnn_hidden_size]) for _ in range(self.num_layers)
            ]
        elif self.rnn_type == "LSTM":
            # Keras expects a flat state list: [h1, c1, h2, c2, ...]
            states = []
            for _ in range(self.num_layers):
                states.append(tf.zeros([batch_size, self.rnn_hidden_size]))  # h
                states.append(tf.zeros([batch_size, self.rnn_hidden_size]))  # c
            return states
    
    def process_step(
        self,
        received_signal,
        beam_index,
        state,
        *,
        step_index=None,
        num_steps=None,
        training=False,
    ):
        """
        Process one sensing step.
        
        Args:
            received_signal: Received complex signal y_t, shape (batch,)
            beam_index: BS beam index x_t, shape (batch,)
            state: RNN hidden state from previous step
            
        Returns:
            combining_vector: Receive beam w_t, shape (batch, num_antennas)
            feedback: Feedback vector m_t, shape (batch, num_feedback)
            new_state: Updated RNN state
        """
        batch_size = tf.shape(received_signal)[0]
        
        # Prepare RNN input
        # Concatenate: [real(y_t), imag(y_t), x_t (one-hot or scalar)]
        y_real = tf.reshape(tf.math.real(received_signal), [batch_size, 1])
        y_imag = tf.reshape(tf.math.imag(received_signal), [batch_size, 1])

        beam_index_i32 = tf.cast(beam_index, tf.int32)
        if self.beam_index_encoding == "one_hot":
            x_feat = tf.one_hot(beam_index_i32, depth=self.codebook_size, dtype=tf.float32)
        else:
            # Normalize beam index to [0, 1] range using actual codebook size
            x_feat = tf.reshape(
                tf.cast(beam_index_i32, tf.float32)
                / tf.cast(self.codebook_size - 1, tf.float32),
                [batch_size, 1],
            )

        features = [y_real, y_imag, x_feat]
        if self.include_time_feature:
            if num_steps is None:
                raise ValueError("num_steps must be provided when include_time_feature=True")
            denom = float(max(int(num_steps) - 1, 1))
            if step_index is None:
                t_norm = 0.0
            else:
                t_norm = float(step_index) / denom
            t_feat = tf.fill([batch_size, 1], tf.constant(t_norm, tf.float32))
            features.append(t_feat)

        rnn_output = tf.concat(features, axis=-1)
        if self.input_ln is not None:
            rnn_output = self.input_ln(rnn_output)

        # Run the stacked recurrent cells for one step.
        if self.rnn_type == "GRU":
            if len(state) != self.num_layers:
                raise ValueError(
                    f"Expected GRU state list of length {self.num_layers}, got {len(state)}."
                )
            new_states = []
            x = rnn_output
            for cell, h_prev in zip(self.rnn_cells, state):
                x, [h_new] = cell(x, [h_prev], training=training)
                new_states.append(h_new)
            rnn_output = x
        else:
            expected = 2 * self.num_layers
            if len(state) != expected:
                raise ValueError(
                    f"Expected LSTM state list of length {expected} (h,c per layer), got {len(state)}."
                )
            new_states = []
            x = rnn_output
            for i, cell in enumerate(self.rnn_cells):
                h_prev = state[2 * i]
                c_prev = state[2 * i + 1]
                x, [h_new, c_new] = cell(x, [h_prev, c_prev], training=training)
                new_states.extend([h_new, c_new])
            rnn_output = x

        if self.output_ln is not None:
            rnn_output = self.output_ln(rnn_output)
        if self.dropout is not None:
            rnn_output = self.dropout(rnn_output, training=training)
        
        # Generate combining vector
        beam_real_imag = self.beam_output(rnn_output)  # (batch, 2*NRX)
        combining_vector = real_to_complex_vector(beam_real_imag, self.num_antennas)
        
        # Normalize beam
        combining_vector = normalize_beam(combining_vector)
        
        # Generate feedback vector m_t
        feedback = self.feedback_output(rnn_output)  # (batch, NFB)

        return combining_vector, feedback, new_states
    
    def call(self, received_signals, beam_indices, initial_state=None, training=False):
        """
        Process a sequence of sensing steps.
        
        Args:
            received_signals: Sequence of received signals, shape (batch, T)
            beam_indices: Sequence of beam indices, shape (batch, T)
            initial_state: Initial RNN state (optional)
            
        Returns:
            combining_vectors: Sequence of receive beams, shape (batch, T, num_antennas)
            feedbacks: Sequence of feedback values, shape (batch, T, num_feedback)
        """
        batch_size = tf.shape(received_signals)[0]
        T = received_signals.shape[1]
        if T is None:
            raise ValueError(
                "UEController.call() requires a statically-known T. "
                "Use BeamAlignmentModel.execute_beam_alignment() for the end-to-end loop."
            )
        
        # Initialize state if not provided
        if initial_state is None:
            state = self.get_initial_state(batch_size)
        else:
            state = initial_state
        
        # Lists to store outputs
        combining_vectors_list = []
        feedbacks_list = []
        
        # Process each time step
        for t in range(int(T)):
            y_t = received_signals[:, t]
            x_t = beam_indices[:, t]

            combining_vector, feedback, state = self.process_step(
                y_t,
                x_t,
                state,
                step_index=t,
                num_steps=int(T),
                training=training,
            )
            
            combining_vectors_list.append(combining_vector)
            feedbacks_list.append(feedback)
        
        # Stack outputs
        combining_vectors = tf.stack(combining_vectors_list, axis=1)  # (batch, T, NRX)
        feedbacks = tf.stack(feedbacks_list, axis=1)  # (batch, T, NFB)
        
        return combining_vectors, feedbacks


if __name__ == "__main__":
    print("Testing UE Controller...")
    print("=" * 60)
    
    # Create UE controller
    ue_controller = UEController(
        num_antennas=16,
        rnn_hidden_size=128,
        rnn_type="GRU",
        num_feedback=4
    )
    
    # Test single step processing
    batch_size = 10
    received_signal = tf.complex(
        tf.random.normal([batch_size]),
        tf.random.normal([batch_size])
    )
    beam_index = tf.constant([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    
    initial_state = ue_controller.get_initial_state(batch_size)
    print(f"Initial state shape: {initial_state.shape if isinstance(initial_state, tf.Tensor) else [s.shape for s in initial_state]}")
    
    combining_vector, feedback, new_state = ue_controller.process_step(
        received_signal, beam_index, initial_state
    )
    
    print(f"\nSingle step output:")
    print(f"  Combining vector shape: {combining_vector.shape}")
    print(f"  Combining vector norm: {tf.norm(combining_vector, axis=-1)[:3]}")
    print(f"  Feedback shape: {feedback.shape}")
    
    # Test sequence processing
    T = 8
    received_signals = tf.complex(
        tf.random.normal([batch_size, T]),
        tf.random.normal([batch_size, T])
    )
    beam_indices = tf.tile(tf.reshape(tf.range(T), [1, T]), [batch_size, 1])
    
    combining_vectors, feedbacks = ue_controller(received_signals, beam_indices)
    
    print(f"\nSequence processing:")
    print(f"  Combining vectors shape: {combining_vectors.shape}")
    print(f"  Feedbacks shape: {feedbacks.shape}")
    print(f"  Beam norms (first sample, all steps): {tf.norm(combining_vectors[0], axis=-1)}")
    
    # Test trainability
    print(f"\nNumber of trainable variables: {len(ue_controller.trainable_variables)}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
