1. **Motivation**: Forecasting clear skies in San Francisco at 500 m resolution is useful and challenging.

For anyone who's spent some amount of time in San Francisco, fog is a force to be reckoned with. It might be 75°F and sunny in Dogpatch, but 60°F and foggy at Ocean Beach. Temperature and cloud cover can change drastically within a few blocks. When planning weekend outings around the city, I’ve therefore become a regular visitor to fog.today. The website displays recent satellite imagery, which you can replay in a 24-hour loop to watch the fog evolve. But I couldn't find a good fine-scale fog forecast ("fogcast") tool.

That’s what motivated this fog forecasting, or fogcasting, project. I wanted a tool that could provide detailed, sub-kilometer-resolution fog predictions for the hours ahead. I couldn’t find one, so I decided to build it. This project is an hourly fogcasting tool, for the next 10 hours following issuance.

2. **Approach**: GOES imagery, numerical weather prediction inputs, and a residual U-Net.



3. **Architecture**: A clean schematic showing how observation and forecast fields feed the network.

Baseline Model A combines current satellite observations, geography and solar geometry with NBM cloud-cover forecasts. Five fine-grid channels enter the encoder; 25 coarse conditioning channels are concatenated with learned features at the 2 km feature level. The model predicts clear-sky probabilities at 500 m for +1 through +10 hours in one forward pass.

The architecture is a residual U-Net with four encoder-to-decoder skip connections. At each decoder stage, features are bilinearly upsampled and concatenated with the corresponding encoder features. Four downsampling residual blocks reduce spatial resolution using a 3×3 convolution with stride 2.

Within each residual block, a parallel residual shortcut bypasses the convolutional layers and is added to the main branch’s output. The shortcut uses an identity mapping when shapes match, or a 1×1 projection to align channels and spatial resolution when needed. After the final decoder, a 1×1 convolution followed by a sigmoid produces ten hourly clear-sky probability maps on the 500 m, 400×400 grid.

![Baseline A residual U-Net: five fine input channels and 25 observation, NBM forecast and time channels merge at 2 km. Four skips, including the post-fusion 64-channel skip, join bilinearly upsampled decoder features. A convolution and sigmoid produce ten hourly clear-sky probability maps at 500 m.](docs/architecture/residual_unet_fogcast_architecture.svg)

4. **Selected technical components**: Links to the implementations and brief explanations of the important design choices.

5. **Results**: Representative predictions compared with observations, persistence, and NBM, with quantitative results where available.

6. Reproduction: Dependencies and instructions for running the published examples.

7. Project status: Active research project; selected components released.

I'd also link to your Fogcast blog posts. The posts can explain why you made certain decisions, while the repository demonstrates how you implemented them.