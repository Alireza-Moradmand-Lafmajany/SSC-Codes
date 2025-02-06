% MATLAB Code for Offline Identification of f(u(k)) using ANFIS
clear;
clc;

% Generate input signal (sinusoid) for training
k_max = 1000; % Total time steps
u_train = sin(2*pi*(1:k_max)/250); % Training input signal u(k)

% Define the unknown nonlinear function f(u)
f = @(u) 0.6*sin(pi*u) + 0.3*sin(3*pi*u) + 0.1*sin(5*pi*u);

% Generate plant output y(k) using the difference equation
y_train = zeros(1, k_max); % Initialize output vector
y_train(1) = 0; % Initial condition y(1)
y_train(2) = 0; % Initial condition y(2)
for k = 2:k_max-1
    y_train(k+1) = 0.3*y_train(k) + 0.6*y_train(k-1) + f(u_train(k)); % Plant dynamics
end

% Prepare training data for ANFIS: [u(k), f(u(k))]
f_u_train = zeros(1, k_max-2); % Initialize f(u(k)) vector
for k = 2:k_max-1
    f_u_train(k-1) = y_train(k+1) - 0.3*y_train(k) - 0.6*y_train(k-1); % Compute f(u(k))
end
input_data_train = u_train(2:k_max-1)'; % Input: u(k)
output_data_train = f_u_train'; % Output: f(u(k))

% Initialize ANFIS using genfis
num_mf = 7; % Number of membership functions for input u(k)
genfis_opt = genfisOptions('GridPartition', 'NumMembershipFunctions', num_mf);
fis = genfis(input_data_train, output_data_train, genfis_opt); % Use all data for initialization

figure;
plotmf(fis, 'input', 1);
title('Initial Membership Functions of Input u(k)');

% Set ANFIS options for offline learning
opt = anfisOptions('InitialFIS', fis, ...
                   'EpochNumber', 200, ... % Number of training epochs
                   'InitialStepSize', 0.1, ...
                   'StepSizeDecreaseRate', 0.9, ...
                   'StepSizeIncreaseRate', 1.1, ...
                   'DisplayANFISInformation', 1, ...
                   'DisplayErrorValues', 1, ...
                   'DisplayStepSize', 1, ...
                   'DisplayFinalResults', 1);

% Train ANFIS offline and capture training error
[fis, trainError] = anfis([input_data_train, output_data_train], opt);

% Predict f(u(k)) for the training input using the trained ANFIS
f_u_train_hat = evalfis(fis, input_data_train);

% Predict the plant output using the identified f(u(k))
y_train_hat = zeros(1, k_max); % Initialize predicted output vector
y_train_hat(1) = 0; % Initial condition y_hat(1)
y_train_hat(2) = 0; % Initial condition y_hat(2)
for k = 2:k_max-1
    y_train_hat(k+1) = 0.3*y_train_hat(k) + 0.6*y_train_hat(k-1) + f_u_train_hat(k-1); % Predicted plant dynamics
end

% Plot results for the training phase
figure;
subplot(2,1,1);
plot(1:k_max, y_train, 'b', 'LineWidth', 1.5); hold on;
plot(1:k_max, y_train_hat, 'r--', 'LineWidth', 1.5);
legend('Actual Output (y)', 'Predicted Output (y\_hat)');
xlabel('Time Step (k)');
ylabel('Output');
title('Training Phase: Actual vs. Predicted Output');

subplot(2,1,2);
plot(1:k_max, y_train - y_train_hat, 'g', 'LineWidth', 1.5);
legend('Prediction Error');
xlabel('Time Step (k)');
ylabel('Error');
title('Training Phase: Prediction Error');

% Plot training error (loss) vs. epochs
figure;
plot(trainError, 'LineWidth', 1.5);
xlabel('Epochs');
ylabel('Training Error (Loss)');
title('Training Error vs. Epochs');

% Plot ANFIS network architecture
figure;
plotfis(fis);
title('ANFIS Network Architecture');

% Plot membership functions of the input before and after training
figure;
plotmf(fis, 'input', 1);
title('Final Membership Functions of Input u(k)');

% Test the trained ANFIS with a new input signal
k_max_test = 1000; % Total time steps for testing
u_test = 0.5*sin(2*pi*(1:k_max_test)/250) + 0.5*sin(2*pi*(1:k_max_test)/25); % Test input signal u(k)

% Generate plant output y(k) for the test input
y_test = zeros(1, k_max_test); % Initialize test output vector
y_test(1) = 0; % Initial condition y(1)
y_test(2) = 0; % Initial condition y(2)
for k = 2:k_max_test-1
    y_test(k+1) = 0.3*y_test(k) + 0.6*y_test(k-1) + f(u_test(k)); % Plant dynamics
end

% Predict f(u(k)) for the test input using the trained ANFIS
f_u_test_hat = evalfis(fis, u_test(2:k_max_test-1)');

% Predict the plant output using the identified f(u(k))
y_test_hat = zeros(1, k_max_test); % Initialize predicted output vector
y_test_hat(1) = 0; % Initial condition y_hat(1)
y_test_hat(2) = 0; % Initial condition y_hat(2)
for k = 2:k_max_test-1
    y_test_hat(k+1) = 0.3*y_test_hat(k) + 0.6*y_test_hat(k-1) + f_u_test_hat(k-1); % Predicted plant dynamics
end

% Plot results for the test input
figure;
subplot(2,1,1);
plot(1:k_max_test, y_test, 'b', 'LineWidth', 1.5); hold on;
plot(1:k_max_test, y_test_hat, 'r--', 'LineWidth', 1.5);
legend('Actual Output (y)', 'Predicted Output (y\_hat)');
xlabel('Time Step (k)');
ylabel('Output');
title('Testing Phase: Actual vs. Predicted Output');

subplot(2,1,2);
plot(1:k_max_test, y_test - y_test_hat, 'g', 'LineWidth', 1.5);
legend('Prediction Error');
xlabel('Time Step (k)');
ylabel('Error');
title('Testing Phase: Prediction Error');