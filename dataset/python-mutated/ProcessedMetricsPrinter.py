import statistics
import pandas as pd
from tabulate import tabulate

class ProcessedMetricsPrinter:

    def print_data_frame(self, name, processed_metrics):
        if False:
            print('Hello World!')
        print(f'metrics for {name}')
        data_frame = self.get_data_frame(processed_metrics)
        print(tabulate(data_frame, showindex=False, headers=data_frame.columns, tablefmt='grid'))

    def combine_processed_metrics(self, processed_metrics_list):
        if False:
            print('Hello World!')
        '\n        A method that merges the value arrays of the keys in the dictionary\n        of processed metrics.\n\n        Args:\n            processed_metrics_list (list): a list containing dictionaries with\n                recorded metrics as keys, and the values are lists of elapsed times.\n\n        Returns::\n            A merged dictionary that is created from the list of dictionaries passed\n                into the method.\n\n        Examples::\n            >>> instance = ProcessedMetricsPrinter()\n            >>> dict_1 = trainer1.get_processed_metrics()\n            >>> dict_2 = trainer2.get_processed_metrics()\n            >>> print(dict_1)\n            {\n                "forward_metric_type,forward_pass" : [.0429, .0888]\n            }\n            >>> print(dict_2)\n            {\n                "forward_metric_type,forward_pass" : [.0111, .0222]\n            }\n            >>> processed_metrics_list = [dict_1, dict_2]\n            >>> result = instance.combine_processed_metrics(processed_metrics_list)\n            >>> print(result)\n            {\n                "forward_metric_type,forward_pass" : [.0429, .0888, .0111, .0222]\n            }\n        '
        processed_metric_totals = {}
        for processed_metrics in processed_metrics_list:
            for (metric_name, values) in processed_metrics.items():
                if metric_name not in processed_metric_totals:
                    processed_metric_totals[metric_name] = []
                processed_metric_totals[metric_name] += values
        return processed_metric_totals

    def get_data_frame(self, processed_metrics):
        if False:
            return 10
        df = pd.DataFrame(columns=['name', 'min', 'max', 'mean', 'variance', 'stdev'])
        for metric_name in sorted(processed_metrics.keys()):
            values = processed_metrics[metric_name]
            row = {'name': metric_name, 'min': min(values), 'max': max(values), 'mean': statistics.mean(values), 'variance': statistics.variance(values), 'stdev': statistics.stdev(values)}
            df = df.append(row, ignore_index=True)
        return df

    def print_metrics(self, name, rank_metrics_list):
        if False:
            for i in range(10):
                print('nop')
        if rank_metrics_list:
            metrics_list = []
            for (rank, metric) in rank_metrics_list:
                self.print_data_frame(f'{name}={rank}', metric)
                metrics_list.append(metric)
            combined_metrics = self.combine_processed_metrics(metrics_list)
            self.print_data_frame(f'all {name}', combined_metrics)

    def save_to_file(self, data_frame, file_name):
        if False:
            while True:
                i = 10
        file_name = f'data_frames/{file_name}.csv'
        data_frame.to_csv(file_name, encoding='utf-8', index=False)