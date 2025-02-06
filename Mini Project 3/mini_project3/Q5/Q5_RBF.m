% Load the data from the Excel file
data = readtable('AirQualityUCI.xlsx');

% Extract the NO2(GT) column as the output
outputData = data.NO2_GT_;

% Extract the input features (all columns except NO2(GT), Date, and Time)
inputData = data{:, setdiff(data.Properties.VariableNames, {'NO2_GT_', 'Date', 'Time'})};

% Handle missing values (replace -200 with NaN)
inputData(inputData == -200) = NaN;
outputData(outputData == -200) = NaN;

% Remove rows with missing values
validRows = ~any(isnan(inputData), 2) & ~isnan(outputData);
inputData = inputData(validRows, :);
outputData = outputData(validRows, :);

% Normalize the input features (z-score normalization)
inputData = normalize(inputData);

% Split the data into training (60%), validation (20%), and testing (20%) sets
rng(42); % Set seed for reproducibility
n = size(inputData, 1);
idx = randperm(n);
trainIdx = idx(1:round(0.6*n));
valIdx = idx(round(0.6*n)+1:round(0.8*n));
testIdx = idx(round(0.8*n)+1:end);

X_train = inputData(trainIdx, :);
Y_train = outputData(trainIdx);

X_val = inputData(valIdx, :);
Y_val = outputData(valIdx);

X_test = inputData(testIdx, :);
Y_test = outputData(testIdx);

% Define the RBF neural network
% The RBF network will have one hidden layer with RBF neurons
% The output layer will be a linear layer

% Number of RBF neurons in the hidden layer
numRBFNeurons = 20;

% Create the RBF network
net = newrb(X_train', Y_train', 0, 5, numRBFNeurons, 1);

% Train the network (RBF networks are trained in one step, no further training is needed)

% Evaluate the network on the validation set
Y_val_pred = sim(net, X_val');

% Calculate the Mean Squared Error (MSE) on the validation set
mse_val = mean((Y_val' - Y_val_pred).^2);
fprintf('Validation MSE: %f\n', mse_val);

% Evaluate the network on the test set
Y_test_pred = sim(net, X_test');

% Calculate the Mean Squared Error (MSE) on the test set
mse_test = mean((Y_test' - Y_test_pred).^2);
fprintf('Test MSE: %f\n', mse_test);

% Plot the test set results
figure;
plot(Y_test, 'b');
hold on;
plot(Y_test_pred, 'r');
legend('Actual', 'Predicted');
xlabel('Sample Index');
ylabel('NO2(GT)');
title('Test Set: Actual vs Predicted');

% Plot the test error (difference between actual and predicted values)
test_error = Y_test' - Y_test_pred;
figure;
plot(test_error, 'k');
xlabel('Sample Index');
ylabel('Test Error');
title('Test Set: Prediction Error');

% Plot a histogram of the NO2(GT) data
figure;
histfit(test_error, 20, 'normal');
xlabel('Error of NO2(GT)');
ylabel('Probability Density');
title('Histogram of Test Error');

% Evaluate the model on the validation set and plot the results
figure;
plot(Y_val, 'b');
hold on;
plot(Y_val_pred, 'r');
legend('Actual', 'Predicted');
xlabel('Sample Index');
ylabel('NO2(GT)');
title('Validation Set: Actual vs Predicted');

% Plot the validation error (difference between actual and predicted values)
val_error = Y_val' - Y_val_pred;
figure;
plot(val_error, 'k');
xlabel('Sample Index');
ylabel('Validation Error');
title('Validation Set: Prediction Error');