import evaluate

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_cer(pred_ids, label_ids, processor):
    """
    Compute the Character Error Rate (CER) between predicted and label sequences.

    Args:
        pred_ids (List[List[int]]): List of predicted token IDs.
        label_ids (List[List[int]]): List of label token IDs.
        processor: The processor object used for decoding.

    Returns:
        float: The computed Character Error Rate (CER).
    """
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

def compute_wer(pred_ids, label_ids, processor):
    """
    Compute the Word Error Rate (WER) between predicted and label sequences.

    Args:
        pred_ids (List[List[int]]): List of predicted token IDs.
        label_ids (List[List[int]]): List of label token IDs.
        processor (transformers.PreTrainedProcessor): 
            Processor object (e.g., from Hugging Face) used for decoding.

    Returns:
        float: The computed Word Error Rate (WER).
    """
    # Validate inputs
    if not isinstance(pred_ids, list) or not isinstance(label_ids, list):
        raise ValueError("Both `pred_ids` and `label_ids` must be lists of token IDs.")

    if not hasattr(processor, "batch_decode"):
        raise ValueError("`processor` must have a `batch_decode` method.")

    # Decode predicted and label sequences
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return wer

def compute_ler(pred_ids, label_ids, processor):
    """
    Compute the Line Error Rate (LER) between predicted and label sequences.

    Args:
        pred_ids (List[List[int]]): List of predicted token IDs.
        label_ids (List[List[int]]): List of label token IDs.
        processor (transformers.PreTrainedProcessor): 
            Processor object (e.g., from Hugging Face) used for decoding.

    Returns:
        float: The computed Line Error Rate (LER).
    """
    # Validate inputs
    if not isinstance(pred_ids, list) or not isinstance(label_ids, list):
        raise ValueError("Both `pred_ids` and `label_ids` must be lists of token IDs.")

    if not hasattr(processor, "batch_decode"):
        raise ValueError("`processor` must have a `batch_decode` method.")

    # Decode predicted and label sequences
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Count line mismatches
    total_lines = len(label_str)
    incorrect_lines = sum(p != l for p, l in zip(pred_str, label_str))

    # Compute LER
    ler = incorrect_lines / total_lines if total_lines > 0 else 0.0

    return ler