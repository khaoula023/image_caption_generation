import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.logger import logging
from src.exception import CustomException
file_path = "artifacts/results_for_Arabic_Flickr8k_3refs.xlsx"
save_path="artifacts/Arabic_Flickr8k_chart_with_values.png"

def plot_metrics_chart(file_path=file_path, save_path=save_path):
    """
    Plot comparison chart for captioning metrics across models.
    """

    try:
        logging.info("Starting metrics plotting pipeline")

        # -----------------------------
        # Load data
        # -----------------------------
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_excel(file_path)
        logging.info("Excel file loaded successfully")

        # -----------------------------
        # Reorder metrics
        # -----------------------------
        metric_order = [
            "BLEU", "METEOR", "CIDEr", "BERTScore",
            "RefCLIPScore", "CLIPScore", "CLAIR",
            "FLEUR", "CHAIR"
        ]

        df = (
            df.set_index("Metric")
              .reindex(metric_order)
              .dropna(how="all")
              .reset_index()
        )

        logging.info("Metrics reordered successfully")

        # -----------------------------
        # Setup plotting
        # -----------------------------
        metrics = df["Metric"]
        models = ["Gemma", "Gemini", "Llama", "Fanar"]

        colors = {
            "Gemma":  "#4C72B0",
            "Gemini": "#DD8452",
            "Llama":  "#55A868",
            "Fanar":  "#8172B2",
        }

        x = np.arange(len(metrics))
        step_width = 0.18
        bar_width = 0.18

        plt.figure(figsize=(13, 4))
        logging.info("Figure initialized")

        # -----------------------------
        # Plot bars
        # -----------------------------
        for i, model in enumerate(models):
            if model not in df.columns:
                logging.warning(f"{model} not found in dataframe, skipping")
                continue

            bars = plt.bar(
                x + i * step_width,
                df[model],
                bar_width,
                label=model,
                color=colors[model],
                edgecolor="black",
                linewidth=0.4
            )

            # Add values
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.5,
                    round(yval, 2),
                    ha='center',
                    va='bottom',
                    fontsize=6
                )

        # -----------------------------
        # X-axis styling
        # -----------------------------
        red_metrics = ["BLEU", "METEOR", "CIDEr", "BERTScore", "RefCLIPScore", "CLAIR"]
        green_metrics = ["CLIPScore", "FLEUR", "CHAIR"]

        x_tick_colors = [
            '#2c7bb6' if m in red_metrics else '#d7191c' if m in green_metrics else 'black'
            for m in metrics
        ]

        plt.xticks(
            x + step_width * (len(models) - 1) / 2,
            metrics,
            ha="center",
            fontsize=10
        )

        ax = plt.gca()
        for ticklabel, tickcolor in zip(ax.get_xticklabels(), x_tick_colors):
            ticklabel.set_color(tickcolor)

        # -----------------------------
        # Grid & legend
        # -----------------------------
        plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

        plt.legend(
            title="Models",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=4,
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            edgecolor="0.6",
            fontsize=11,
            title_fontsize=12
        )

        # Clean look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Adjust y-limit
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax * 1.1)

        plt.tight_layout()

        # -----------------------------
        # Save figure
        # -----------------------------
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

        logging.info(f"Chart saved successfully at {save_path}")

        plt.show()

    except Exception as e:
        logging.error(f"Error in plotting metrics chart: {e}", exc_info=True)
        raise CustomException(e, sys)