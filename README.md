Q1: Basic Autoencoder for Image Reconstruction
Goal: Learn compressed representations of images and reconstruct them.

Architecture: Fully connected (Dense) encoder → latent space → decoder.

Dataset: MNIST (28x28 images → flattened to 784).

Loss: Binary Cross-Entropy.

Insight: Smaller latent dimensions = higher compression, but lower reconstruction quality.

Q2: Denoising Autoencoder
Goal: Reconstruct clean images from noisy inputs.

Noise: Added using np.random.normal(mean=0, std=0.5).

Training: Input = noisy image, Target = original clean image.

Benefit: Robustness to corrupted inputs; useful in real-world applications like medical imaging.

Q3: Text Generation with RNN (LSTM)
Goal: Predict the next character in a sequence.

Data: Text corpus (e.g., Shakespeare), converted to character indices.

Model: LSTM layer + Dense (softmax over characters).

Sampling: Generate text character-by-character using temperature:

Low temp = deterministic, repetitive

High temp = creative, diverse, but possibly incoherent

Q4: Sentiment Classification Using LSTM
Goal: Classify text reviews as positive or negative.

Dataset: IMDB (pre-tokenized reviews).

Preprocessing: Tokenization + Padding.

Model: Embedding → LSTM → Dense(sigmoid).

Evaluation: Confusion matrix, precision, recall, F1-score.

Precision-Recall Tradeoff:

Important when false positives/negatives have different costs (e.g., mislabeling a bad review as good).
