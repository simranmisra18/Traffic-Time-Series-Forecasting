# PERIODIC TIME SERIES PREDICTION USING CONVOLUTIONAL NEURAL NETWORK

## Model Description
In this project, a Convolutional Neural Network (CNN) model is designed for 1D time-series data and aims to capture
 temporal patterns through a series of convolutional and pooling operations.
### Dataset Preparation
 The dataset was preprocessed using a sliding window technique to generate training samples, implemented in the
 TrajectoryDataset class. This custom transformation creates a series of overlapping windows from the raw data. Each
 window sequence serves as an input sample, with the last point in the sequence designated as the target. This technique
 helps the model learn patterns within a moving time frame, which is essential for accurate time-series predictions.
 
 ### Model Architecture
 The model is defined as a custom class CNNModel in PyTorch, with the following structure:
 1. Convolution Layer: The model starts with a 1D convolutional layer (self.conv1) with a kernel size of 2, which
 allows it to process small segments of the input time-series data. This layer uses 64 filters, and the kernel size
 is chosen to capture local temporal dependencies.

 2. ReLU Activation: After the convolution, a ReLU activation function (self.relu) is applied to introduce
 non-linearity, which helps the model learn complex patterns.

3. Max Pooling Layer: The Max Pooling layer (self.pool) reduces the dimensionality of the data by taking the
 maximum value within each window of size 2, reducing the data size by half and making the model more
 computationally efficient.

 4. Dropout Layer: A dropout rate of 0.3 is applied to prevent overfitting by randomly deactivating neurons during
 training.

 5. Fully Connected Layer: Flattened outputs from the convolutional layers are passed to a fully connected layer
 to produce a single output, predicting the next trajectory point.

 6. Forward Pass: The input is normalized using nn.functional.normalize to stabilize the training process. We
 then add a channel dimension to transform the input to a shape compatible with CNN layers. The output is
 flattened for input into the fully connected layer, producing the final output predictions.

 ### Training Procedure
 The model was trained using the following settings:
 
 1. Loss Function: Mean Squared Error (MSE) was used to measure the prediction error.
 2. Optimizer: Adam optimizer with a learning rate of 0.001 is used, which adjusts weights based on gradients,
 aiming for efficient convergence.
 3. Learning Rate Scheduler: ReduceLROnPlateau2 scheduler is used that reduces the learning rate by a factor of
 0.1 when there is no improvement in validation loss for 10 epochs, helping avoid local minima.
 4. Epochs and Batch Size: The model was trained for one epoch with a batch size of 32.
 3 Validation and Model Selection
 The model’s performance was evaluated on a separate validation set by performing autoregressive predictions. The
 autoregressive prediction function is implemented to predict a series of future values, one step at a time. The function
 iteratively takes in the model’s output, that is the predicted values, as part of the input for subsequent predictions. This
 autoregressive approach is beneficial for sequential data like time series, where each time step depends on previous
 predictions. The validation method included computing the Mean Squared Error (MSE) between the predicted trajectory
 and the actual data points from the validation set. Using MSELoss() in PyTorch, the computed MSE quantifies the
 prediction accuracy of the autoregressive model, with lower MSE indicating better performance.

 ## Results
 Weachieved the final training loss of 0.0013, and the final MSE for the validation is 0.0018, which would indicate a low
 error on validation data. This score indicates the model’s effectiveness in capturing trajectory trends, making it suitable
 for testing on unseen data, which would be the test data. Below are the predicted versus actual trajectories for the three
 trajectory instances (0, 1, 2) from the validation dataset. Figure 1 shows the plots that illustrate the model’s accuracy in
 predicting trajectory paths over time and provide insight into its ability to track the true trajectory of traffic flow data.
 Each plot was saved as a PNG file with dimensions 4x4 inches and a resolution of 200 DPI
