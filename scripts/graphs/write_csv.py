import numpy as np
import csv
import random
import matplotlib.pyplot as plt

filename = "experiments.csv"
latency_rowname = "Latency (ms)"
tput_rowname = "Throughput (queries/s)"
acorn_qps_row = "ACORN Queries per microsecond"
oak_qps_row = "OAK Queries per microsecond"
sub_qps_row = "SUB Queries per microsecond"
acorn_recall_row = "ACORN Recall@10"
oak_recall_row = "OAK Recall@10"
sub_recall_row = "SUB Recall@10"
def make_fake_csv():
    np.random.seed(42)  
    throughput = np.linspace(100, 1000, 50)
    latency = 1 / (throughput / 100) + np.random.normal(0, 0.1, throughput.size)
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([latency_rowname, tput_rowname])

        for i in range(len(throughput)):
            writer.writerow([latency[i], throughput[i]])


def plot(output: str, data: str):
    # Lists to store data from the CSV
    acorn_throughput = []
    acorn_recall = []
    oak_throughput =[]
    oak_recall=[]
    sub_throughput = []
    sub_recall = []

    # Read the CSV file
    with open(data, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            acorn_throughput.append(float(row[acorn_qps_row]))
            acorn_recall.append(float(row[acorn_recall_row]))
            oak_throughput.append(float(row[oak_qps_row]))
            oak_recall.append(float(row[oak_recall_row]))
            sub_throughput.append(float(row[sub_qps_row]))
            sub_recall.append(float(row[sub_recall_row]))
    # Plot the graph
    plt.figure(figsize=(10, 6)) 
    plt.plot(acorn_throughput, acorn_recall, marker='o', color='b', label="ACORN")  
    plt.plot(oak_throughput, oak_recall, marker='o', color='g', label="OAK")
    plt.plot(sub_throughput, sub_recall, marker='o', color='y', label="SUB")

    # Add labels and title
    plt.xlabel("Throughput (requests per millisecond)") 
    plt.ylabel("Recall@10")  
    plt.title("Latency vs Throughput")
    plt.legend()
    plt.grid(True)

    plt.savefig(output)
    # Show the plot
    plt.show() 

plot("latency_throughput_plot.png", filename)