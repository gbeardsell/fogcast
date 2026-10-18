Your README will probably be read by more people than your code.
I'd structure it around the following:
1. Motivation: Why forecasting clear skies in San Francisco at 500 m resolution is useful and challenging.
2. Approach: GOES imagery, numerical weather prediction inputs, and a residual U-Net.
3. Architecture: A clean schematic showing how observation and forecast fields feed the network.
4. Selected technical components: Links to the implementations and brief explanations of the important design choices.
5. Results: Representative predictions compared with observations, persistence, and NBM, with quantitative results where available.
6. Reproduction: Dependencies and instructions for running the published examples.
7. Project status: Active research project; selected components released.
I'd also link to your Fogcast blog posts. The posts can explain why you made certain decisions, while the repository demonstrates how you implemented them.