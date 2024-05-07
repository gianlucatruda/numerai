# Gianluca's Numerai models and experiments


From May 2021 to Jan 2023, I competed as a ML/Data Scientist on [Numerai](https://numer.ai/), the "hardest data science tournament in the world".

- Each week, I generated stock market predictions using several ML models I’ve developed.
- I staked those models with my own money (using Numerai's cryptocurrency, Numeraire).
- My predictions contributed to the Numerai meta-model which drives their hedge fund positions. 
- I also played around with generating additional [trading signals](https://signals.numer.ai/) by scraping data from other sources, like [prediction markets](https://www.metaculus.com).

I got pretty good at it:
- I regularly placed in the top-100 staked models.
- My best model earned 95% annual return in 2022 (even while the global markets were eating shit).

I learned a ton about:
- algorithmic trading
- quantitative finance
- building performant models from challenging tabular datasets

But in late 2022, I started my own AI company and couldn't devote any more time to improving my models. Numerai has also been expanding their underlying dataset, so I would have had to spend a lot of development time to update and expand my models. I decided to stop competing weekly.

So I've decided to open-source my code as a jumping off point from others who are interesting in learning more about Numerai, data science, and quantitative trading.

## What you'll find 

I did everything in Jupyter notebooks that I ran in Google Colab. 

This was great for the experimentation process and for being able to trigger my models to automatically run and submit predictions via a web browser from anywhere in the world. But in general, it's not the best setup: Jupyter notebooks are hard to version and maintain and although Colab is great for getting access to decent GPUs for free/cheap, it's pretty janky. 

```
.
├── README.md
├── experiments
│   ├── Numerai development v3_21.ipynb
│   └── Numerai signals development v0.2.ipynb
└── models
    ├── Numerai GTRUDA stable.ipynb
    └── Numerai V3X stable.ipynb
```

- `experiments/` contains the notebooks where I analysed the data, tried new feature engineering techniques, and compared my new model ideas against various baselines (including my own best-performing model, `V3X`).
- `models/` contains the notebooks that I would run each week to submit predictions from my current "stable" models:
    - `V3X` was my best-performing model over time, based on a meta-ensemble of gradient boosted trees. It never made a killing, but it was super consistent and robust.
    - `GTRUDA` was a neural network based on a convolutional autoencoder. It did really well some weeks, but was a bit unstable in training and had high variance. But developing that model inspired many ideas behind my later research on [TableDiffusion](http://gianluca.ai/table-diffusion).


## Tips for getting started with Numerai

1. Read the [numerai docs](https://docs.numer.ai/), they're excellent and will help you get started fast.
2. Play around and build your own simple models. You can submit weekly predictions from them for free without needing to stake any money.
3. Get really good by analysing and understanding the dataset. Note that the dataset has updated from the version I was using back in 2022.
4. Start staking your own models with small sums that you're happy to lose. Get skin in the game.
5. Come back here to get ideas for other approaches to try and model architecture inspiration. 
6. Check out the [Numerai forum](https://forum.numer.ai/) and bounce ideas with others. Most people are doing this for fun, so they're willing to share information and code.


