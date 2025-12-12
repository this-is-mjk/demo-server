from typing import Dict, Tuple

def aggregate_predictions(model_outputs: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    """
    Decides the final diagnosis based on the weighted average of all model outputs.
    """
    aggregated_scores = {}
    
    # Sum up probabilities from all models
    for model_name, predictions in model_outputs.items():
        for disease, score in predictions.items():
            aggregated_scores[disease] = aggregated_scores.get(disease, 0) + score

    # Find the disease with the highest accumulated score
    best_disease = max(aggregated_scores, key=aggregated_scores.get)
    
    # Calculate an average confidence score (approximate)
    num_models = len(model_outputs)
    avg_confidence = aggregated_scores[best_disease] / num_models

    return best_disease, avg_confidence
