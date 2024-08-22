def asr_metric(eval_result):
    loss_weight = 0.7
    wer_weight = 0.3
    normalized_loss = eval_result["eval_loss"] / 10
    return loss_weight * normalized_loss + wer_weight * eval_result["eval_wer"]
