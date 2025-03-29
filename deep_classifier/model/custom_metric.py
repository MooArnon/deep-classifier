import tensorflow as tf

class BalancedLabelMetric(tf.keras.metrics.Metric):
    def __init__(
        self, 
        name="balanced_label_metric", 
        alpha=0.5, 
        beta1=1.0,
        **kwargs,
    ) -> None:
        """
        Custom metric for binary classification:
         - class 0 = SHORT
         - class 1 = LONG

        We combine the F1-scores of each class in a weighted way:
            prb = alpha * F1(LONG) + (1 - alpha) * F1(SHORT)
        And define Composite Financial Performance Index (CFPI):
            cfpi = beta1 * prb

        Then average CFPI across batches.
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta1 = beta1

        # We track sum of CFPI and batch count
        self.cfpi_sum = self.add_weight(name="cfpi_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_pred.shape[-1] == 1:  
            y_pred_labels = tf.cast(y_pred >= 0.5, tf.int32)
        else:
            y_pred_labels = tf.argmax(y_pred, axis=1)  

        y_true = tf.cast(y_true, tf.int32)

        f1_scores = []
        num_classes = 2  

        for class_idx in range(num_classes):
            true_positives = tf.reduce_sum(
                tf.cast((y_true == class_idx) & (y_pred_labels == class_idx), tf.float32)
            )
            predicted_positives = tf.reduce_sum(
                tf.cast(y_pred_labels == class_idx, tf.float32)
            )
            actual_positives = tf.reduce_sum(
                tf.cast(y_true == class_idx, tf.float32)
            )

            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            recall = true_positives / (actual_positives + tf.keras.backend.epsilon())

            f1_score = 2 * (precision * recall) / (
                precision + recall + tf.keras.backend.epsilon()
            )

            f1_scores.append(f1_score)

        f1_short = f1_scores[0]
        f1_long  = f1_scores[1]

        # Compute PRB and CFPI
        prb = self.alpha * f1_long + (1.0 - self.alpha) * f1_short
        cfpi = self.beta1 * prb

        # Accumulate
        self.cfpi_sum.assign_add(cfpi)
        self.count.assign_add(1.0)

    def result(self):
        """Return the average CFPI across all batches so far."""
        return self.cfpi_sum / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        """Reset the accumulators."""
        self.cfpi_sum.assign(0.0)
        self.count.assign(0.0)
