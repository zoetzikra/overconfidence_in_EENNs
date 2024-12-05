import os  

def read_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Skip the header
    data = [line.strip().split('\t') for line in lines[1:]]
    return data

def calculate_statistics(data):
    total_samples = len(data)
    correct_count = sum(int(row[2]) for row in data)
    correct_percentage = (correct_count / total_samples) * 100

    exit_point_counts = {}
    incorrect_exit_point_3_count = 0
    incorrect_count = 0

    for row in data:
        exit_point = int(row[1])
        correct = int(row[2])

        if exit_point not in exit_point_counts:
            exit_point_counts[exit_point] = 0
        exit_point_counts[exit_point] += 1

        if not correct:
            incorrect_count += 1
            if exit_point == 3:
                incorrect_exit_point_3_count += 1

    return correct_percentage, exit_point_counts, incorrect_count, incorrect_exit_point_3_count

def append_statistics(file_path, correct_percentage, exit_point_counts, incorrect_count, incorrect_exit_point_3_count):
    with open(file_path, 'a') as file:
        file.write("\nStatistics:\n")
        file.write(f"Percentage of data classified correctly: {correct_percentage:.2f}%\n")
        file.write("Data points per exit point:\n")
        for exit_point, count in exit_point_counts.items():
            file.write(f"  Exit Point {exit_point}: {count}\n")
        file.write(f"Incorrect data points with exit point 3: {incorrect_exit_point_3_count}\n")

def main():
    file_path = 'test_results.txt'
    data = read_results(file_path)
    correct_percentage, exit_point_counts, incorrect_exit_point_3_count = calculate_statistics(data)
    append_statistics(file_path, correct_percentage, exit_point_counts, incorrect_exit_point_3_count)

if __name__ == "__main__":
    main()