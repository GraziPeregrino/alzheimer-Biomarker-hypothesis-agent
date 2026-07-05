"""
annotation_server.py — a thin MCP server for variant annotation.

This does NOT reimplement any annotation logic. It wraps the project's existing
`skills.ingestion_skills.annotate_variant` (curated knowledge base + cached
Ensembl / MyVariant / GWAS Catalog enrichment) behind the Model Context Protocol,
so any MCP client — including the ADK/Gemini agent — can call it as a standardized
"literature / genetic lookup" tool.

Run standalone (stdio transport):

    python mcp_servers/annotation_server.py
"""
import os
import sys

# This file runs as a subprocess script, so sys.path[0] is this folder — add the
# project root so `skills.*` imports resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mcp.server.fastmcp import FastMCP

# The existing annotation functions — wrapped, not rewritten.
from skills.ingestion_skills import annotate_variant as _annotate_variant, VARIANT_KB

mcp = FastMCP("biomarker-annotation")


@mcp.tool()
def annotate_variant(query: str, use_network: bool = True) -> dict:
    """Annotate an Alzheimer's genetic variant or gene with curated facts plus
    optional live enrichment from MyVariant.info / Ensembl / GWAS Catalog (cached
    to disk, so repeat and offline calls still succeed).

    Args:
        query: rsID ('rs429358'), gene ('APOE', 'TREM2'), or alias ('APOE e4', 'R47H').
        use_network: If False, return only curated + previously-cached data.
    """
    return _annotate_variant(query, use_network=use_network)


@mcp.tool()
def known_variants() -> dict:
    """List the curated variants the annotator recognizes (rsID, gene, AD relevance)."""
    return {"variants": [
        {"rsid": rsid, "gene": entry.get("gene"),
         "ad_relevance": entry.get("ad_relevance")}
        for rsid, entry in VARIANT_KB.items()
    ]}


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
