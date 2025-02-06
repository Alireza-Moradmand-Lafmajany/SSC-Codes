clc;
clear all;
close all;

% Load the data from the .dat file
data = load('ballbeam.dat');
U = data(:, 1); % Input: angle of the beam
Y = data(:, 2); % Output: position of the ball

% Combine input and output data for ANFIS
inputData = [U, Y];

% Split the data into training and testing sets
numSamples = size(inputData, 1);
numTrain = round(0.75 * numSamples); % 75% for training
trainData = inputData(1:numTrain, :);
testData = inputData(numTrain+1:end, :);

% Normalize the data
trainData = normalize(trainData);
testData = normalize(testData);

% Generate initial FIS structure
numMFs = 3; % Number of membership functions
fis = genfis1(trainData, numMFs, 'gaussmf');

figure;
plotmf(fis, 'input', 1);
title('Initial Membership Functions of Input');

% Train ANFIS model and capture training error
epoch_n = 300; % Number of epochs
[fis, trainError] = anfis(trainData, fis, epoch_n);

% Plot the training error vs epochs
figure;
plot(1:epoch_n, trainError, 'b', 'LineWidth', 1.5);
xlabel('Epochs');
ylabel('Training Error');
title('Training Error vs Epochs');
grid on;

figure;
plotmf(fis, 'input', 1);
title('Final Membership Functions of Input')

% Test the ANFIS model
testInput = testData(:, 1);
testOutput = testData(:, 2);
predictedOutput = evalfis(testInput, fis);

% Calculate the test error
testError = testOutput - predictedOutput;

% Plot the test results and test error in subplots
figure;
subplot(2, 1, 1);
plot(testOutput, 'b', 'LineWidth', 1.5);
hold on;
plot(predictedOutput, 'r', 'LineWidth', 1.5);
legend('Actual Output', 'Predicted Output');
xlabel('Sample');
ylabel('Position of the Ball');
title('ANFIS Prediction vs Actual Output');
grid on;

subplot(2, 1, 2);
plot(testError, 'm', 'LineWidth', 1.5);
xlabel('Sample');
ylabel('Test Error');
title('Test Error');
grid on;

% Plot a histogram of the test error with a fitted distribution
figure;
histfit(testError, 20, 'normal'); % 20 bins, normal distribution fit
xlabel('Test Error');
ylabel('Frequency');
title('Histogram of Test Error with Fitted Distribution');
grid on;

% Calculate and display the RMSE
rmse = sqrt(mean(testError.^2));
fprintf('Root Mean Squared Error (RMSE): %f\n', rmse);