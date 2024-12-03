from Ex2_main_algorithms import *

# Main algorithm parameters
initial_tau = 0.1
final_time = 3
x_discretization = 50
y_discretization = 50

# Processing parameters
plot_frequency = 50
should_report_quantity_loss = True

transport2d(
    initial_tau,
    final_time,
    x_discretization,
    y_discretization,
    plot_frequency,
    should_report_quantity_loss,
)
