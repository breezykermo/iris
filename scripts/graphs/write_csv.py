import numpy as np
import csv
import random
import matplotlib.pyplot as plt

filename = "latency_throughput_data.csv"
latency_rowname = "Latency (ms)"
tput_rowname = "Throughput (queries/s)"
def make_fake_csv():
    np.random.seed(42)  
    throughput = np.linspace(100, 1000, 50)
    latency = 1 / (throughput / 100) + np.random.normal(0, 0.1, throughput.size)
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([latency_rowname, tput_rowname])

        for i in range(len(throughput)):
            writer.writerow([latency[i], throughput[i]])


def plot():
    # Lists to store data from the CSV
    latency = []
    throughput = []

    # Read the CSV file
    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            latency.append(float(row[latency_rowname]))
            throughput.append(float(row[tput_rowname]))
    # Plot the graph
    plt.figure(figsize=(10, 6)) 
    plt.plot(throughput, latency, marker='o', color='b', label="Latency vs Throughput")  

    # Add labels and title
    plt.xlabel("Throughput (requests per second)") 
    plt.ylabel("Latency (ms)")  
    plt.title("Latency vs Throughput")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show() 

plot()