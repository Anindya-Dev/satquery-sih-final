"""Router engine implementation for SatQuery SIH pipeline."""

class TaskRouterEngine:
    def route_dict(self, user_query: str) -> dict:
        query_lower = user_query.lower()
        
        # Match primary tool based on query intent
        if "flood" in query_lower or "inundation" in query_lower:
            primary_tool = "sar_flood_extractor"
            target = "flood"
        elif "fusion" in query_lower or ("sar" in query_lower and "optical" in query_lower):
            primary_tool = "optical_sar_fusion_specialist"
            target = "fusion"
        elif "grounding" in query_lower or "locate" in query_lower or "built-up" in query_lower:
            primary_tool = "grounding_rs_specialist"
            target = "grounding"
        elif "change" in query_lower or "bitemporal" in query_lower:
            primary_tool = "bitemporal_change_detector"
            target = "change"
        else:
            primary_tool = "spectral_indices_calculator"
            target = "spectral"

        return {
            "primary_tool": primary_tool,
            "target": target,
            "query": user_query
        }

def route(task):
    engine = TaskRouterEngine()
    return engine.route_dict(str(task))