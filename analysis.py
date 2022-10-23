import numpy as np


WIDTH = 10


def performance(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> dict:
    if len(predicted_labels) != len(actual_labels):
        raise Exception(f"labels must have equal lengths (currently {len(predicted_labels)} != {len(actual_labels)})")

    data = {"accuracy": np.mean(predicted_labels == actual_labels)}

    unique_labels = list(np.unique(actual_labels))

    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), int)
    for i in range(len(actual_labels)):
        confusion_matrix[unique_labels.index(predicted_labels[i])][unique_labels.index(actual_labels[i])] += 1

    data["precision"], data["recall"], data["f1-score"], data["support"] = {}, {}, {}, {}
    for label in unique_labels:
        l_idx = unique_labels.index(label)
        data["precision"][label] = confusion_matrix[l_idx][l_idx] / np.sum(confusion_matrix, axis=1)[l_idx]
        data["recall"][label] = confusion_matrix[l_idx][l_idx] / np.sum(confusion_matrix, axis=0)[l_idx]
        data["f1-score"][label] = (
            2 * data["precision"][label] * data["recall"][label] / (data["precision"][label] + data["recall"][label])
        )
        data["support"][label] = np.sum(confusion_matrix, axis=0)[l_idx]

    data["labels"] = unique_labels

    return data


def report(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> str:
    data = performance(predicted_labels, actual_labels)

    report_str, just = "", max(len(label) for label in data["labels"]) + 1
    report_str += (
        f"{'precision'.rjust(WIDTH + just)}{'recall'.rjust(WIDTH)}{'f1-score'.rjust(10)}{'support'.rjust(WIDTH)}\n\n"
    )

    for label in data["labels"]:
        report_str += f"{label.ljust(just)}"
        for metric in ("precision", "recall", "f1-score", "support"):
            report_str += (
                f"{data[metric][label]:.5f}".rjust(WIDTH)
                if metric != "support"
                else f"{data[metric][label]}".rjust(WIDTH)
            )
        report_str += "\n"

    report_str += f"\n{'accuracy'.ljust(just)}"
    report_str += "".rjust(WIDTH * 2)
    report_str += f"{data['accuracy']}".rjust(WIDTH)
    report_str += f"{sum(val for val in data['support'].values())}".rjust(WIDTH)

    return report_str
