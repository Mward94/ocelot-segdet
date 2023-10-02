import os
from typing import Optional, Iterable, Dict, Any

import torch
from tqdm import tqdm

from networks.postprocessors.postprocessor import Postprocessor
from util.meters import MeanValueMeter
from util.torch import move_data_to_device, get_current_lr, save_model_state


def train_model(model, optimiser, train_loader, val_loader, device, num_epochs,
                output_directory, criterion_object=None, scheduler=None, metrics_object=None,
                train_logger=None, val_logger=None, criterion_train_only=True,
                metrics_val_only=True, validation_frequency=1,
                validation_behaviour='single',
                postprocessors: Optional[Iterable[Postprocessor]] = None):
    """Performs training of a DL model

    This handles performing the training and validation passes over the model for a given number of
    epochs

    Args:
        model: Trainable DL model
        optimiser (Optimiser): The optimiser used for training
        train_loader (DataLoader): The PyTorch dataloader to get batches from when training
        val_loader (DataLoader/None): The PyTorch dataloader to get batches from when validating.
            If None, no validation performed
        device (torch.device): The device to train on (cpu/gpu)
        num_epochs (int): The number of epochs to train for
        output_directory (str): Path to the directory to store model weights in
        criterion_object (Object/None): A HistoCriterion object that computes the loss given model
            outputs and ground truth. When compute() is called, should return loss value that can be
            back-propagated. Can be returned as single valur or as a dictionary of loss_name:
            loss_value (supporting multiple losses), If None, assumes the model returns the loss
            when training
        scheduler (optim.lr_scheduler/None): The learning rate scheduler (if using) or None. If
            present, the scheduler will be stepped after **each epoch**.
        metrics_object (Object/None): A HistoMetric object that can track and compute metrics for
            outputs from a batch. When compute() is called, should return a dictionary with
            keys=name of metric and values = value of metric of metrics computed on ALL data passed
            during all calls of update()
        train_logger (CSVLogger/None): The CSV logger object to log training epoch metrics to
        val_logger (CSVLogger/None): The CSV logger object to log validation epoch metrics to
        criterion_train_only (bool): If True, the given criterion function is ONLY used for the
            training pass, if False is used for both training and validation
        metrics_val_only (bool): If True, the given metric class is ONLY used for the validation
            pass, if False is used for both training and validation
        validation_frequency (int): How often (in epochs) a pass over the validation set should
            occur
        validation_behaviour (str): The behaviour of how validation should be performed. Supports
            either 'single' or 'reconstruct'. When set to 'single', metrics are computed per-sample
            on the validation set and globally aggregated. When set to 'reconstruct',
            outputs/ground truth are first reconstructed to the entire region, then metrics are
            computed comparing the reconstructed output/entire ground truth. These per-region
            metrics are then globally aggregated. When setting 'reconstruct', it is important that
            the dataset is tiled. 'reconstruct' behaviour is slower, but more closely resembles how
            metrics are computed on the test set. 'single' behaviour is faster, but may not be
            truly representative of the test set.
        postprocessors: Postprocessors to apply to the model output. If None, the default model
            postprocessors will be used.
    """
    # Ensure valid validation_behaviour
    if validation_behaviour not in ('single', 'reconstruct'):
        raise ValueError(f'Validation behaviour {validation_behaviour} not supported.')

    # Setup whether metrics/loss is also computed for training/validation
    train_metrics_object = None if metrics_val_only else metrics_object
    val_criterion_object = None if criterion_train_only else criterion_object

    # Ensure model is on the correct device
    model.to(device)

    print(f'Training for {num_epochs} epochs')
    for epoch_num in range(num_epochs):
        # Determine if a pass over the validation set will take place
        pass_over_val = (val_loader is not None
                         and validation_frequency
                         and (epoch_num + 1) % validation_frequency == 0)

        # Train for an epoch
        train_losses, train_metrics, current_lr = train_epoch(
            model, optimiser, train_loader, device, epoch_num,
            criterion_object=criterion_object, metrics_object=train_metrics_object,
            logger=train_logger, scheduler=scheduler, postprocessors=postprocessors)

        # Validate for an epoch
        if pass_over_val:
            val_losses, val_metrics = eval_epoch(
                model, val_loader, device, epoch_num, criterion_object=val_criterion_object,
                metrics_object=metrics_object, logger=val_logger,
                reconstruct_output=validation_behaviour == 'reconstruct',
                postprocessors=postprocessors)

        if output_directory is not None:
            # Save the model state
            model_latest_path = os.path.join(output_directory, f'model-latest.pth')
            save_model_state(model_latest_path, model, epoch_num=epoch_num, optimiser=optimiser,
                             scheduler=scheduler)


def train_epoch(model, optimiser, loader, device, epoch_num, criterion_object=None,
                metrics_object=None, logger=None, scheduler=None,
                postprocessors: Optional[Iterable[Postprocessor]] = None):
    """Trains the model for one epoch across the dataloader

    Args:
        model: Trainable DL model
        optimiser (Optimiser): The optimiser
        loader (DataLoader): The PyTorch dataloader to get batches from
        device (torch.device): The device to train on (cpu/gpu)
        epoch_num (int): The current epoch number
        criterion_object (Object/None): A HistoCriterion object that computes the loss given model
            outputs and ground truth. When compute() is called, should return loss value that can be
            back-propagated. Can be returned as single valur or as a dictionary of loss_name:
            loss_value (supporting multiple losses), If None, assumes the model returns the loss
            when training
        metrics_object (Object/None): A HistoMetric object that can track and compute metrics for
            outputs from a batch. When compute() is called, should return a dictionary with
            keys=name of metric and values = value of metric of metrics computed on ALL data passed
            during all calls of update()
        logger (CSVLogger/None): The CSV logger object to log epoch metrics to
        scheduler (LRScheduler/None): The learning rate scheduler.
        postprocessors: Postprocessors to apply to the model output.

    Returns:
        epoch_losses (dict): A dictionary with keys = loss names and value = mean loss value
        epoch_metrics (dict): A dictionary with keys = metric names and value = mean metric value
        learning_rate (float): The current learning rate after training for an epoch
    """
    # Setup stores for loss across epoch
    #   This is updated by example in batch (i.e. loss computed with batch_size of 4 counts as 4)
    #   This means when computing the average, it is the average per-example
    epoch_losses = {'epoch_average_loss': MeanValueMeter()}

    # Put model into training mode
    model.train()

    # Extract the keys the model needs from the batch for training
    model_input_keys = model.required_input_keys

    # Iterate through batches in the dataloader
    progress = tqdm(loader, leave=True)
    for batch_idx, batch in enumerate(progress):
        # Create a store of all data put onto the GPU for this batch (to not duplicate GPU data)
        batch_gpu_data = {}

        # Extract required inputs for the model from the batch (and move onto device)
        model_inputs = {key: move_data_to_device(batch[key], device) for key in model_input_keys}

        # Update the store of data on the GPU
        batch_gpu_data.update(model_inputs)

        # Clear gradient buffers
        optimiser.zero_grad()

        # Perform the forward pass
        model_outputs = model.forward(model_inputs)

        # Check that we have all the model outputs required by the criterion
        criterion_keys = set(criterion_object.model_output_keys)
        missing_keys = criterion_keys.difference(model_outputs.keys())
        if len(missing_keys) > 0:
            raise RuntimeError('The model does not provide all of the outputs required '
                               f'for the criterion. Missing keys: {missing_keys}')

        # Extract required model outputs for criterion object
        criterion_model_output = {key: model_outputs[key] for key in criterion_keys}

        # Extract required ground truth for criterion class (put onto device as required)
        criterion_ground_truth = {}
        for key in criterion_object.ground_truth_keys:
            if key in batch_gpu_data:
                criterion_ground_truth[key] = batch_gpu_data[key]
            else:
                criterion_ground_truth[key] = move_data_to_device(batch[key], device)
                batch_gpu_data[key] = criterion_ground_truth[key]

        # Compute the loss
        loss = criterion_object.compute(criterion_model_output, criterion_ground_truth)

        # Sum up losses (in case multiple losses returned)
        total_loss = sum(loss_value for loss_value in loss.values())

        # Back propagate loss
        total_loss.backward()

        # Step optimiser (update model weights)
        optimiser.step()

        # Update metrics with results from batch (call update() method on metric object)
        if metrics_object is not None:
            # Perform any postprocessing on outputs
            if model_outputs is not None:
                model_outputs = postprocess_outputs(model_outputs, postprocessors)

            # Extract required model outputs for metrics class
            metrics_model_output = {key: move_data_to_device(model_outputs[key], device)
                                    for key in metrics_object.model_output_keys}

            # Extract required ground truth for metrics class (put onto device as required)
            metrics_ground_truth = {}
            for key in metrics_object.ground_truth_keys:
                if key in batch_gpu_data:
                    metrics_ground_truth[key] = batch_gpu_data[key]
                else:
                    metrics_ground_truth[key] = move_data_to_device(batch[key], device)
                    batch_gpu_data[key] = metrics_ground_truth[key]

            # Update the metrics object with data from this batch
            running_metrics = metrics_object.update(metrics_model_output, metrics_ground_truth)
        else:
            running_metrics = None

        # ############################## Update metrics for the epoch ##############################
        # Determine the number of samples in the batch (to weight overall loss contribution)
        #   Do this based on the length of one of the input keys given to the model
        num_inputs = len(batch[model_input_keys[0]])

        # Epoch average loss
        epoch_losses['epoch_average_loss'].add(total_loss, num_inputs)

        # Individual Losses
        for loss_name, loss_value in loss.items():
            if loss_name in epoch_losses:
                epoch_losses[loss_name].add(loss_value, num_inputs)
            else:
                epoch_losses[loss_name] = MeanValueMeter()
                epoch_losses[loss_name].add(loss_value, num_inputs)

        # Create a metrics string to display
        if running_metrics is not None:
            running_metrics_str = '. Metrics - ' + ', '.join(
                [f'{metric_name}: ' + (f'{metric_value:.4f}' if metric_value is not None else 'None')
                 for metric_name, metric_value in running_metrics.items()])
        else:
            running_metrics_str = ''

        # Update label on progress bar (limit metrics string length to 40 characters)
        progress.set_description(f'Epoch {epoch_num}: Training - Average epoch loss: '
                                 f'{epoch_losses["epoch_average_loss"].get_mean():.6f}'
                                 f'{running_metrics_str[:40]}')

    # Per-epoch scheduler step.
    scheduler.step()

    # Get the whole epoch metrics, then reset the metrics object
    if metrics_object is not None:
        epoch_metrics = metrics_object.compute()
        metrics_object.reset()
    else:
        epoch_metrics = {}

    # Get the current learning rate (to be logged)
    current_lr = get_current_lr(optimiser)

    if logger is not None:
        # Log data for the epoch to file
        data_to_log = {
            'epoch': epoch_num,
            **{loss_name: value_meter.get_mean() for loss_name, value_meter in epoch_losses.items()},
            **{metric_name: metric_value for metric_name, metric_value in epoch_metrics.items()},
            'lr': current_lr,
        }
        logger.write_dict(data_to_log)

    return {loss_name: loss_meter.get_mean() for loss_name, loss_meter in epoch_losses.items()}, \
           {metric_name: metric_value for metric_name, metric_value in epoch_metrics.items()}, current_lr


def eval_epoch(model, loader, device, epoch_num=0, criterion_object=None, metrics_object=None,
               logger=None, reconstruct_output=False,
               postprocessors: Optional[Iterable[Postprocessor]] = None):
    """Evaluates the model for one epoch across the dataloader

    Args:
        model: DL model
        loader (DataLoader): The PyTorch dataloader to get batches from
        device (torch.device): The device to put data on (cpu/gpu)
        epoch_num (int): The current epoch number
        criterion_object (Object/None): A HistoCriterion object that computes the loss given model
            outputs and ground truth. When compute() is called, should return loss value that can be
            back-propagated. Can be returned as single valur or as a dictionary of loss_name:
            loss_value (supporting multiple losses), If None, assumes the model returns the loss
            when training. This is only used for loss logging. If not given, no losses will be
            logged
        metrics_object (Object/None): A HistoMetric object that can track and compute metrics for
            outputs from a batch. When compute() is called, should return a dictionary with
            keys=name of metric and values = value of metric of metrics computed on ALL data passed
            during all calls of update()
        logger (CSVLogger/None): The CSV logger object to log epoch metrics to
        reconstruct_output (bool): If True, the model outputs will first be reconstructed before
            they are fed to the metric class. When specifying this as True, it is absolutely
            critical that the dataset is not shuffled, and regions appear in order
        postprocessors: Postprocessors to apply to the model output.

    Returns:
        epoch_losses (dict): A dictionary with keys = loss names and value = mean loss value
        epoch_metrics (dict): A dictionary with keys = metric names and value = mean metric value
    """
    # Setup stores for loss across epoch
    #   This is updated by example in batch (i.e. loss computed with batch_size of 4 counts as 4)
    #   This means when computing the average, it is the average per-example
    epoch_losses = {}

    # Put model into evaluation mode
    model.eval()

    # Extract the keys the model needs from the batch for evaluation
    model_input_keys = model.required_input_keys

    # Create a store for model predictions and coordinates if reconstruct_output is True
    if reconstruct_output and metrics_object is not None:
        region_data_store = {}
        loaded_region_paths = []

    # Iterate through batches in the dataloader
    progress = tqdm(loader, leave=True)
    progress.set_description(f'Epoch {epoch_num}: Evaluating')
    for batch_idx, batch in enumerate(progress):
        # Setup the region data store (if required)
        if reconstruct_output and metrics_object is not None:
            # Determine if this is the last batch (to compute final metrics)
            last_batch = batch_idx == len(loader) - 1

            # Set up the region_data_store
            num_samples = len(batch['input_path'])
            for sample_idx in range(num_samples):
                input_path = batch['input_path'][sample_idx]
                if input_path not in region_data_store:
                    dimensions = batch['dimensions'][sample_idx]
                    if isinstance(dimensions, torch.Tensor):
                        dimensions = dimensions.cpu().numpy()
                    region_data_store[input_path] = {
                        'dimensions': dimensions,
                        'outputs': [],
                        'output_coords': [],
                    }
                    loaded_region_paths.append(input_path)

                # Append 'prediction_coords' for each sample to the region_data_store
                output_coordinates = batch['output_coordinates'][sample_idx]
                if isinstance(output_coordinates, torch.Tensor):
                    output_coordinates = output_coordinates.cpu().numpy()
                output_coordinates = output_coordinates.tolist()
                region_data_store[input_path]['output_coords'].append(output_coordinates)

        # Create a store of all data put onto the GPU for this batch (to not duplicate GPU data)
        batch_gpu_data = {}

        # Extract required inputs for the model from the batch (and move onto device)
        model_inputs = {key: move_data_to_device(batch[key], device) for key in model_input_keys}

        # Update the store of data on the GPU
        batch_gpu_data.update(model_inputs)

        # Perform forward pass
        with torch.no_grad():
            model_outputs = model.forward(model_inputs)

        # Compute the loss (if a criterion object is given)
        if criterion_object is not None:
            criterion_model_output = {key: model_outputs[key]
                                      for key in criterion_object.model_output_keys}

            # Extract required ground truth for criterion class (put onto device as required)
            criterion_ground_truth = {}
            for key in criterion_object.ground_truth_keys:
                if key in batch_gpu_data:
                    criterion_ground_truth[key] = batch_gpu_data[key]
                else:
                    criterion_ground_truth[key] = move_data_to_device(batch[key], device)
                    batch_gpu_data[key] = criterion_ground_truth[key]

            # Compute the loss
            loss = criterion_object.compute(criterion_model_output, criterion_ground_truth)

        # Update metrics with results from batch (call update() method on metric object)
        running_metrics = None
        if metrics_object is not None:
            # Move model outputs to CPU
            model_outputs = {k: move_data_to_device(v, torch.device('cpu'))
                             for k, v in model_outputs.items()}

            if not reconstruct_output:
                # Perform any postprocessing on outputs
                # Since we are not reconstructing output, we do not collate outputs first
                model_outputs = postprocess_outputs(model_outputs, postprocessors)

                # Extract required model outputs for metrics class
                metrics_model_output = {key: model_outputs[key]
                                        for key in metrics_object.model_output_keys}

                # Extract required ground truth for metrics class (put onto device as required)
                metrics_ground_truth = {}
                for key in metrics_object.ground_truth_keys:
                    metrics_ground_truth[key] = move_data_to_device(batch[key], torch.device('cpu'))

                # Update the metrics object with data from this batch
                running_metrics = metrics_object.update(metrics_model_output, metrics_ground_truth)
            else:
                # Update the store with output predictions
                for sample_idx in range(num_samples):
                    input_path = batch['input_path'][sample_idx]

                    # Update the predictions store for that model with all model outputs
                    region_data_store[input_path]['outputs'].append({
                        key: model_outputs[key][sample_idx]
                        for key in model_outputs.keys()
                    })

                # Update the metrics object if we have complete data for one of the inputs
                while len(loaded_region_paths) > 1 or last_batch:
                    # Extract one of the complete predictions
                    path_to_compute = loaded_region_paths.pop(0)
                    region_data = region_data_store.pop(path_to_compute)

                    # Request a new complete input/ground truth
                    complete_input_gt = loader.dataset.get_complete_input_and_ground_truth_for_input_path(
                        path_to_compute)

                    # Reconstruct all model outputs into a single output
                    complete_output = model.collate_outputs(
                        region_data['outputs'], region_data['output_coords'],
                        region_data['dimensions'])

                    # Perform any postprocessing on outputs
                    # Since we are reconstructing output, this must happen *after* collation
                    complete_output = postprocess_outputs(complete_output, postprocessors)

                    # Extract the ground truth/model output that the metric class requires
                    # IMPORTANT: Given this is reconstructed, we force the ground truth to be on
                    #            the CPU, to avoid running out of memory
                    metrics_ground_truth = {key: move_data_to_device(complete_input_gt[key], torch.device('cpu'))
                                            for key in metrics_object.ground_truth_keys}
                    metrics_model_output = {key: complete_output[key]
                                            for key in metrics_object.model_output_keys}

                    # Given we are updating output-by-output, create a 'batch' dimension
                    for key in metrics_ground_truth:
                        if isinstance(metrics_ground_truth[key], torch.Tensor):
                            metrics_ground_truth[key] = metrics_ground_truth[key].unsqueeze(0)
                        else:
                            metrics_ground_truth[key] = [metrics_ground_truth[key]]
                    for key in metrics_model_output:
                        if isinstance(metrics_model_output[key], torch.Tensor):
                            metrics_model_output[key] = metrics_model_output[key].unsqueeze(0)
                        else:
                            metrics_model_output[key] = [metrics_model_output[key]]

                    # Update metrics object
                    running_metrics = metrics_object.update(metrics_model_output, metrics_ground_truth)

                    # Stop iteration if this was triggered due to the last batch
                    if len(loaded_region_paths) == 0 and last_batch:
                        break

        # ############################# Update criterion for the epoch #############################
        if criterion_object is not None:
            # Determine the number of samples in the batch (to weight overall loss contribution)
            #   Do this based on the length of one of the input keys given to the model
            num_inputs = len(batch[model_input_keys[0]])

            for loss_name, loss_value in loss.items():
                if loss_name in epoch_losses:
                    epoch_losses[loss_name].add(loss_value, num_inputs)
                else:
                    epoch_losses[loss_name] = MeanValueMeter()
                    epoch_losses[loss_name].add(loss_value, num_inputs)

        # Update progress message
        if running_metrics is not None:
            # Update the progress bar with running metric information
            progress_str = ' - ' + ', '.join(
                [f'{metric_name}: ' + (f'{metric_value:.4f}' if metric_value is not None else 'None')
                 for metric_name, metric_value in running_metrics.items()])
            progress.set_description(f'Epoch {epoch_num}: Evaluating{progress_str[:40]}')

    # Get the whole epoch metrics, call write(), then reset the metrics object
    if metrics_object is not None:
        epoch_metrics = metrics_object.compute()
        metrics_object.reset()
    else:
        epoch_metrics = {}

    if logger is not None:
        # Log data for the epoch to file
        data_to_log = {
            'epoch': epoch_num,
            **{loss_name: value_meter.get_mean() for loss_name, value_meter in epoch_losses.items()},
            **{metric_name: metric_value for metric_name, metric_value in epoch_metrics.items()},
        }
        logger.write_dict(data_to_log)

    return {loss_name: loss_meter.get_mean() for loss_name, loss_meter in epoch_losses.items()}, \
           {metric_name: metric_value for metric_name, metric_value in epoch_metrics.items()}


def postprocess_outputs(
        outputs: Dict[str, Any], postprocessors: Optional[Iterable[Postprocessor]],
) -> Dict[str, Any]:
    """Postprocess model outputs.

    The model outputs will be passed through a chain of postprocessors and the final result
    returned. Additionally, the model output keys will be validated to ensure that each
    postprocessor has all the data it needs.

    Args:
        outputs: The original model outputs.
        postprocessors: A list of postprocessors to apply to the outputs (in order).

    Returns:
        The postprocessed model outputs.
    """
    if postprocessors is None:
        return outputs

    for postprocessor in postprocessors:
        actual_keys = set(outputs.keys())
        required_keys = set(postprocessor.model_output_keys)
        key_diff = required_keys - actual_keys
        if len(key_diff) > 0:
            key_diff_str = ', '.join(sorted([k for k in key_diff]))
            raise ValueError(f'The {postprocessor.__class__.__name__} postprocessor requires the '
                             f'following missing model outputs: {key_diff_str}')
        outputs = postprocessor.postprocess(outputs)

    return outputs
