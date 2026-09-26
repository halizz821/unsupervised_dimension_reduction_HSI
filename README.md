# Usupervised dimension reduction for hyperspectral images

This is an implementation of the weighted feature extraction (WFE) and fuzzy feature extraction (FFE) feature extraction techniques that we proposed in our paper below. If you use it, kindly cite it.

@Article{rs15153855,
AUTHOR = {Alizadeh Moghaddam, Sayyed Hamed and Gazor, Saeed and Karami, Fahime and Amani, Meisam and Jin, Shuanggen},
TITLE = {An Unsupervised Feature Extraction Using Endmember Extraction and Clustering Algorithms for Dimension Reduction of Hyperspectral Images},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {15},
ARTICLE-NUMBER = {3855},
URL = {https://www.mdpi.com/2072-4292/15/15/3855},
ISSN = {2072-4292},
DOI = {10.3390/rs15153855}
}


- Run test.m to see how used the functions.

## Acknowledgements
This project utilizes the [Open Source MATLAB Hyperspectral Toolbox] [https://github.com/isaacgerg/matlabHyperspectralToolbox] developed by [Isaac Gerg] [https://github.com/isaacgerg]. This toolbox played a crucial role in implementing my codes within this project.



<script src="https://mermaid.live/embed.js" async></script>
<mermaid-embed src="https://mermaid.live/embed?theme=redux-dark-color&look=handDrawn&mode=dark" height="480">
flowchart TD
    %% Tier 1: Macro Filtering
    subgraph Tier1 [Tier 1: Macro Screening — SentinelAgent]
        A[ECCC Severe Weather Alerts&lt;br/>Live MCP Server or Simulated] --> B[SentinelAgent Scanner&lt;br/>scan_national_portfolio]
        C[(SQLite: insurance_portfolio.db&lt;br/>properties, policies)] --> B
        B --> D[Peril Classifier&lt;br/>Filters property-threatening perils only]
        D --> E[Spatial Correlation Engine&lt;br/>GeoJSON Point-in-Polygon &amp; Bounding Box]
        E --> F[at_risk_candidates.json&lt;br/>Prioritized AtRiskPropertyCandidate Queue]
    end

    %% Tier 2: Micro Deep Dive
    subgraph Tier2 [Tier 2: Micro Deep Dive — Property Mitigation Specialist]
        F --> G[Pipeline Runner / Queue Dispatcher&lt;br/>run_pipeline.py]
        G --> H[agent_reasoner&lt;br/>Gemini 2.5 Flash ReAct Reasoner]
        
        %% Cyclic ReAct Loop
        H &lt;-->|tool_calls / observations| I[tool_node&lt;br/>• tool_get_property_details&lt;br/>• tool_get_policy_coverage&lt;br/>• tool_get_alerts_near_coordinates&lt;br/>• tool_search_alerts]
        
        %% Advisory Formulation
        H -->|Tool gathering complete| J[advisory_formulator&lt;br/>Working Memory Scratchpad +&lt;br/>Structured AdvisoryPayload]
        
        %% LLM Safety Guardrail
        J --> K[safety_guardrail&lt;br/>LLM Safety Auditor with Structured Output&lt;br/>SafetyAuditResult]
        
        %% Self-Correction Reflection Loop
        K -- "Safety Violation Detected&lt;br/>(Self-Correcting Reflection Loop, max 3)" --> H
        K -- "Verified Safe" --> L[dispatch_node&lt;br/>• Formats SMS &amp; Push Notifications&lt;br/>• Commits to SQLite mitigation_dispatches&lt;br/>• Overwrites output/advisories.json]
    end

</mermaid-embed>
