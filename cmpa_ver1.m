%param define:
is = 0.01 * 10^(-12);
ib = 0.1 * 10^(-12);
vb = 1.3;
gp = 0.1;
V = linspace(-1.95, 0.7, 200)';

%calculate I and I with noise (I_noise):
for i=1:200
    I(i) = model(is, ib, vb, gp, V(i));
    r = -0.2 + 0.4*rand();
    I_noise(i) = I(i) + r*I(i);
end

%calculate polyfit() fitting
poly4 = polyfit(V, I_noise, 4); % 6th order polyfit
poly8 = polyfit(V, I_noise, 8); % 8th order polyfit

%fit()
fo1 = fitoptions('Method','NonlinearLeastSquares', ...
                'StartPoint',[is,gp,ib,vb]);
ftype1 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)','options',fo1);
ff1= fit(V, I_noise', ftype1);
fo2 = fitoptions('Method','NonlinearLeastSquares', ...
                'StartPoint',[1,1]);
ftype2 = fittype('A.*(exp(1.2*x/25e-3)-1) + (0.1).*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)','options',fo2);
ff2= fit(V, I_noise', ftype2);
fo3 = fitoptions('Method','NonlinearLeastSquares', ...
                'StartPoint',[1,1,1]);
ftype3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)','options',fo3);
ff3= fit(V, I_noise', ftype3);

%neural net fitting
Vnn = V;
Inn = zeros(200, 1);
inputs = Vnn.';
targets = I;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn_out = outputs;

%plotting
figure(1);
subplot(2, 2, 1);
plot(V, I_noise);
title("V / I noise");
xlabel("V");
ylabel("I noise");

subplot(2, 2, 2);
semilogy(V, abs(I_noise));
title("V / I noise (semilog)");
xlabel("V");
ylabel("I noise");

subplot(2, 2, 3);
plot(V, polyval(poly4, V));
hold on;
plot(V, polyval(poly8, V));
plot(V, I_noise);
title("Polyfit() result");
xlabel("V");
ylabel("I noise");
legend("4th order polyfit", "8th order polyfit", "Real data");

subplot(2, 2, 4);
plot(V, ff1(V));
hold on;
plot(V, ff2(V));
plot(V, ff3(V));
plot(V, I_noise);
title("fit() result");
xlabel("V");
ylabel("I noise");
legend('fit A,B,C,D', 'fit A,C', 'fit A,B,C', 'Real data');

figure(2);
plot(V, Inn_out);
title("NN result");
xlabel("V");
ylabel("I noise");



function diodeI = model(is, ib, vb, gp, V)
    diodeI = is*(exp(1.2*V / 0.025) - 1) + gp*V - ib*(exp(-1.2*(V + vb) / 0.025) - 1);
end