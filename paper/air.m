
%
% Air Throughput Results for Gpu Fixed on ARM Tegra
%

th_thread1 = [15 14 14.7 14.47 22.3 25.2 20];
th_thread3 = [21.5 20 20 16 34.5 32.76 26];
figure;
hold on;
grid on;
plot(th_thread1,'-rs','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5]);
plot(th_thread3,'-bs','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5]);
legend('ARM 1 Thread','ARM 3 Thread','Position',[150,350,1,1])
xlabel('LDPC codes','FontSize',12,'FontWeight','bold') % x-axis label
ylabel('Air throughput (\it{Mbps})','FontSize',12,'FontWeight','bold') % y-axis label