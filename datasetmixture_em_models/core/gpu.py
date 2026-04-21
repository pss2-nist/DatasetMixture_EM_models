"""GPU utilization monitoring and logging."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import GPUtil


def write_header(output_filename):
    fieldnames = ['Epoch', 'Time stamp', 'ID', 'Name', 'Serial', 'UUID', 'GPU temp. [C]', 'GPU util. [%]', 'Memory util. [%]',
                  'Memory total [MB]', 'Memory used [MB]', 'Memory free [MB]', 'Display mode', 'Display active']
    output_dir = os.path.dirname(output_filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def record(epoch: int, output_filename: str) -> None:
    """Record GPU metrics to CSV file."""
    output_dir = os.path.dirname(output_filename)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    GPUs = GPUtil.getGPUs()
    print('INFO: GPUs:', GPUs)
    if not GPUs:
        print('WARNING: the hardware does not contain NVIDIA GPU card')
        return

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y:%m:%d:%H:%M:%S")
    print('INFO: date_time ', date_time)

    attrList = [[{'attr': 'id', 'name': 'ID'},
                 {'attr': 'name', 'name': 'Name'},
                 {'attr': 'serial', 'name': 'Serial'},
                 {'attr': 'uuid', 'name': 'UUID'}],
                [{'attr': 'temperature', 'name': 'GPU temp.', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0},
                 {'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0},
                 {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100,
                  'precision': 0}],
                [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
                 {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0},
                 {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}],
                [{'attr': 'display_mode', 'name': 'Display mode'},
                 {'attr': 'display_active', 'name': 'Display active'}]]

    # store the date_time as teh first entry in the recorded row
    store_gpu_info = str(epoch) + ',' + date_time

    for attr_group in attrList:
        for attr_dict in attr_group:
            precision = f".{attr_dict['precision']}" if "precision" in attr_dict else ""
            transform = attr_dict.get("transform", lambda x: x)

            for gpu in GPUs:
                attr = getattr(gpu, attr_dict["attr"])
                attr = transform(attr)

                if isinstance(attr, float):
                    attr_str = f"{attr:{precision}f}"
                elif isinstance(attr, int):
                    attr_str = str(attr)
                elif isinstance(attr, str):
                    attr_str = attr
                else:
                    raise TypeError(f"Unhandled type {type(attr)} for attribute '{attr_dict['name']}'")

                store_gpu_info += f",{attr_str}"

    store_gpu_info += '\n'
    print('row data:', store_gpu_info)
    with open(output_filename, 'a', newline='') as csvfile:
        csvfile.write(store_gpu_info)


def start_monitoring(output_filename: str, epochs: int = 4, interval: int = 1) -> None:
    """Record GPU metrics at regular intervals for multiple epochs."""
    write_header(output_filename)
    for epoch in range(1, epochs + 1):
        record(epoch, output_filename)
        if epoch < epochs:
            import time
            time.sleep(interval)
