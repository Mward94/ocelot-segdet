from typing import List, Tuple, Type, Optional

import torchmetrics

from util.constants import SEG_MASK_INT, GT_SEG_MASK


class SegMetrics:
    """Computes common segmentation and classification metrics.
    """
    def __init__(
            self,
            class_list: List[str],
            IoU_avg: Optional[List[str]] = None,
            F1_avg: Optional[List[str]] = None,
            recall_avg: Optional[List[str]] = None,
            accuracy_avg: Optional[List[str]] = None,
            precision_avg: Optional[List[str]] = None,
            specificity_avg: Optional[List[str]] = None,
            ignore_class: Optional[str] = None,
    ):
        """Creates the set of torchmetrics objects based on the input parameters

        Args:
            class_list: A list of class names. Indexes of names are used to map indexes of classes
                when returning metrics. It is also required to properly compute the per-class
                metrics.
            IoU_avg: If the IoU (Jaccard index) metric needs to be computed. The metric accepts a
                list of average methods including the following ('macro', 'micro', 'perclass').
            F1_avg: If the F1 or dice metric needs to be computed. The metric accepts a list of
                average methods including the following ('macro', 'micro', 'perclass').
            recall_avg: If the recall or sensitivity metric needs to be computed. The
                metric accepts a list of average methods including the following ('macro', 'micro',
                'perclass').
            accuracy_avg: If the accuracy metric needs to be computed. The metric
                accepts a list of average methods including the following ('macro', 'micro',
                'perclass').
            precision_avg: If the precision metric needs to be computed. The metric
                accepts a list of average methods including the following ('macro', 'micro',
                'perclass').
            specificity_avg: If the specificity metric needs to be computed. The metric
                accepts a list of average methods including the following ('macro', 'micro',
                'perclass').
            ignore_class: Name of a class (as defined in class_list) to ignore from computation.
                Ignored classes are excluded from the averages.
        """
        super().__init__()

        num_classes = len(class_list)

        # Ensure at least 1 average method is specified
        if not any((IoU_avg, F1_avg, recall_avg, accuracy_avg, precision_avg, specificity_avg)):
            raise ValueError('Please specify at least one type of average to compute')

        self.class_list = class_list
        if ignore_class is not None and self.class_list is None:
            raise ValueError('Ignore class is specified without the class list')

        # Set up the ignore_index
        if ignore_class is None:
            self.ignore_index = None
        else:
            self.ignore_index = self.class_list.index(ignore_class)

        self.all_metrics_objects = []

        if IoU_avg is not None:
            self.all_metrics_objects.extend(
                self._create_torch_metric_object(
                    num_classes=num_classes, avg_methods=IoU_avg, ignore_index=self.ignore_index,
                    metric_class=torchmetrics.JaccardIndex, metric_name='IoU'))

        if F1_avg is not None:
            self.all_metrics_objects.extend(
                self._create_torch_metric_object(
                    num_classes=num_classes, avg_methods=F1_avg, ignore_index=self.ignore_index,
                    metric_class=torchmetrics.F1Score, metric_name='F1'))

        if recall_avg is not None:
            self.all_metrics_objects.extend(self._create_torch_metric_object(
                num_classes=num_classes, avg_methods=recall_avg, ignore_index=self.ignore_index,
                metric_class=torchmetrics.Recall, metric_name='Recall'))

        if accuracy_avg is not None:
            self.all_metrics_objects.extend(self._create_torch_metric_object(
                num_classes=num_classes, avg_methods=accuracy_avg, ignore_index=self.ignore_index,
                metric_class=torchmetrics.Accuracy, metric_name='Accuracy'))

        if precision_avg is not None:
            self.all_metrics_objects.extend(self._create_torch_metric_object(
                num_classes=num_classes, avg_methods=precision_avg, ignore_index=self.ignore_index,
                metric_class=torchmetrics.Precision, metric_name='Precision'))

        if specificity_avg is not None:
            self.all_metrics_objects.extend(self._create_torch_metric_object(
                num_classes=num_classes, avg_methods=specificity_avg,
                ignore_index=self.ignore_index, metric_class=torchmetrics.Specificity,
                metric_name='Specificity'))

    @property
    def model_output_keys(self) -> List[str]:
        return [SEG_MASK_INT]

    @property
    def ground_truth_keys(self) -> List[str]:
        return [GT_SEG_MASK]

    def update(self, outputs, ground_truths):
        """Method used to update the internal confusion matrix for the metrics.

        Outputs and ground truths are expected to be passed in as a batch

        This method should be called for each batch.
        This method also returns metrics computed for this batch.

        Args:
            outputs (dict): A dictionary containing all model outputs with keys as defined in
                self.model_output_keys. Values associated to keys in outputs should have shape
                (BxNcxHxW), where B = batch, Nc = number of classes, and H,W = spatial resolution.
                An argmax is applied across the classes dimension to get the predicted class
                per-pixel
            ground_truths (dict): A dictionary containing all ground truth data with keys as
                defined in self.ground_truth_keys. Values associated to keys in ground_truths
                should have shape (BxHxW), where B = batch, and H,W = spatial resolution.

        Returns:
            (dict): A dictionary of the running computed metrics. Keys = metric name,
                values = metric
        """
        pred = outputs[SEG_MASK_INT]
        for _, _, metric_object in self.all_metrics_objects:
            metric_object.to(pred.device)
            metric_object(pred, ground_truths[GT_SEG_MASK])
        return self.compute()

    def compute(self):
        """ Method used to compute the final metrics using the already updated confusion matrixes.

        Returns:
             (dict): A dictionary containing all computed metric values. Keys = name of metric,
                Values = value of metric
        """
        metric_dict = {}
        for (metric_name, avg_method, metric_object) in self.all_metrics_objects:
            value = metric_object.compute().cpu().numpy()
            metric_dict.update(self._create_metric_dict(avg_method, metric_name, value))
        return metric_dict

    def _create_metric_dict(self, avg_method, metric_name, value):
        """Method used to put the computed metric value into a dictionary

        Args:
            avg_method (str): averaging methods used: micro, macro, None (per class)
            metric_name (str): the name of the metric.
            value (float/np[float]): metric value computed by the metric object

        Returns:
            (dict): A dictionary of computed metrics. Keys = metric name, values = metric
        """
        metric_return_dict = {}
        if avg_method == 'perclass':
            # adjustment needed since the IoU metric removes the ignore class and outputs a smaller
            # result vector
            index_adjustment = 0
            for i, class_name in enumerate(self.class_list):
                # Code to handle per-class metrics
                if i != self.ignore_index:
                    metric_return_dict[f'{metric_name} {class_name}'] = value[i - index_adjustment]
                elif metric_name == 'IoU':
                    index_adjustment = 1
        else:
            metric_return_dict[f'{metric_name} {avg_method}'] = value.item()
        return metric_return_dict

    def reset(self):
        """Resets any internal stores so that metric instances can be reused.
        """
        for (_, _, metric_object) in self.all_metrics_objects:
            metric_object.reset()

    def _create_torch_metric_object(
        self,
        num_classes: int,
        avg_methods: List[str],
        ignore_index: int,
        metric_class: Type['Metric'],
        metric_name: str,
    ) -> List[Tuple[str, str, 'Metric']]:
        """Creates torch metric object

        Args:
            num_classes: Number of classes.
            avg_methods: Averaging methods used: micro, macro, perclass (none).
            ignore_index: The class index to ignore during the computation.
            metric_class: The torchmetrics class to use for creating the metric object.
            metric_name: The name of the metric.

        Returns:
            (tuple): A three element tuple (metric name, averaging method, pytorch metric object)
        """
        metric_objects = []
        for avg_method in avg_methods:
            parameter_avg_method = avg_method
            if avg_method == 'perclass':
                parameter_avg_method = 'none'
            elif avg_method not in {'macro', 'micro'}:
                raise ValueError(f'Invalid {metric_name} averaging method specified')
            metric_objects.append((metric_name, avg_method, metric_class(
                num_classes=num_classes, average=parameter_avg_method, ignore_index=ignore_index,
                mdmc_average='global')))
        return metric_objects
