from typing import Any, Dict, List, Union


class VLMSynthesizer:
    """
    Member 4 Module: Ingests Member 3's Evidence list contract and synthesizes 
    a natural language response using verbatim scalar metrics.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path

    async def generate_grounded_answer(
        self,
        user_query: str,
        task_spec: Dict[str, Any],
        evidence: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> str:
        # Handle unsupported task specs or dictionary fallbacks
        if isinstance(evidence, dict):
            if evidence.get("status") == "unsupported":
                note = evidence.get("note", "Requested task is currently unsupported.")
                return f"Unable to process query: {note}"
            entry = evidence
            results = evidence
            target = evidence.get("target", "unknown")
        elif isinstance(evidence, list) and len(evidence) > 0:
            entry = evidence[0]
            results = entry.get("results", {})
            target = entry.get("target") or task_spec.get("parameters", {}).get("target_features", "specified region")
        else:
            return "Unable to process query: Evidence pipeline returned empty results."

        # Extract scalar metrics safely from results dictionary without fabricating fallbacks
        confidence = results.get("confidence_mean", results.get("confidence"))
        
        # Format confidence string or explicitly label as N/A if absent
        if confidence is not None and isinstance(confidence, (int, float)):
            conf_str = f"{confidence * 100:.1f}%" if confidence <= 1.0 else f"{confidence}%"
        elif confidence is not None:
            conf_str = str(confidence)
        else:
            conf_str = "N/A"

        # Dynamically locate area metrics in results
        area_km2 = (
            results.get("flooded_area_km2")
            or results.get("changed_area_km2")
            or results.get("water_area_km2")
            or results.get("built_up_area_km2")
        )
        
        # Check for index or grounding results
        mean_ndvi = results.get("mean_NDVI")
        mean_ndwi = results.get("mean_NDWI")
        bboxes = results.get("bounding_boxes")

        # Build grounded response strictly using scalar metrics
        if area_km2 is not None:
            return (
                f"Based on satellite analysis for target '{target}', "
                f"the system detected an affected coverage of {area_km2} km² "
                f"with a confidence score of {conf_str}."
            )
        elif mean_ndvi is not None:
            return (
                f"Spectral analysis for target '{target}' yields a mean NDVI of {mean_ndvi:.3f} "
                f"with a confidence score of {conf_str}."
            )
        elif mean_ndwi is not None:
            return (
                f"Spectral analysis for target '{target}' yields a mean NDWI of {mean_ndwi:.3f} "
                f"with a confidence score of {conf_str}."
            )
        elif bboxes is not None:
            num_regions = results.get("num_regions", len(bboxes))
            conf_suffix = f" with a confidence score of {conf_str}." if conf_str != "N/A" else "."
            return f"Grounding analysis for target '{target}' identified {num_regions} region(s){conf_suffix}"

        tool_name = entry.get("tool", task_spec.get("primary_tool", "analysis tool"))
        conf_suffix = f" Confidence score: {conf_str}." if conf_str != "N/A" else ""
        return f"Analysis completed via '{tool_name}' for target '{target}'.{conf_suffix}"