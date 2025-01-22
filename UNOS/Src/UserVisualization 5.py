# import plot libries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def countPlot(data, catList, txt='', figsize=(10, 4), labelFontSize=10, titleFontSize=11, annotationFontSize = 11, legend=False):
    # matplotlib figure
    num_cols = len(catList)
    fig_height = figsize[1] * num_cols  # Adjust height based on number of categories
    plt.figure(figsize=(figsize[0], fig_height))
    
    # iterate each categorical column and create a subplot
    for i, column in enumerate(catList, start=1):
        plt.subplot(num_cols, 1, i)  # create a subplot for each category
        ax = sns.countplot(data=data, y=column, hue=column, legend=legend)
        ax.grid(False)  # remove grid
        
        # customize each plot
        plt.title(f"Count Plot for {column} {txt}", fontsize=titleFontSize)
        plt.xlabel("Count", fontsize=labelFontSize)
        plt.ylabel(column, fontsize=labelFontSize)
    
        # display counts on top of each bar
        for p in ax.patches:  # iterate over the bars
            ax.annotate(f'{int(p.get_width()):,}', 
                        (p.get_width() + 100, p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', fontsize=annotationFontSize)
    
    # adjust the vertical space between subplots
    plt.subplots_adjust(hspace=0.4)

    # show the plot
    plt.tight_layout()
    plt.show()



# def countPlot(data, catList, txt=''):
#     # eet up the matplotlib figure
#     num_cols = len(catList)
#     plt.figure(figsize=(10, 5 * num_cols))  # adjust the figure size as needed
    
#     # iterate each categorical column and create a subplot
#     for i, column in enumerate(catList, start=1):
#         plt.subplot(num_cols, 1, i)  # create a subplot for each category
#         ax = sns.countplot(data=data, y=column, hue=column, legend=False)
#         ax.grid(False) # remove grid
        
#         # customize each plot
#         plt.title(f'Count Plot for {column} {txt}')
#         plt.xlabel('Count')
#         plt.ylabel(column)
    
#         # display counts on top of each bar
#         for p in ax.patches:  # iterate over the bars
#             ax.annotate(f'{int(p.get_width()):,}', 
#                         (p.get_width() + 100, p.get_y() + p.get_height() / 2), 
#                         ha='left', va='center')
    
#     # adjust the vertical space between subplots
#     plt.subplots_adjust(hspace=0.4)

#     # Show the plot
#     plt.tight_layout()
#     plt.show()


def histogramPlot(data, lst, bins=30, txt='', titleFont=15, labelFont=10, tickFont=10, KDE=True):
    """
    The function histogramPlot is well-designed for plotting histograms of multiple columns in a single figure. 
    """
    # calculate the number of rows needed (two plots per row)
    num_cols = min(len(lst), 2)  # max 2 columns
    num_rows = int(np.ceil(len(lst) / num_cols))  # calculate number of rows
    
    # set up the matplotlib figure
    plt.figure(figsize=(10 * num_cols, 5 * num_rows))  # adjust the figure size as needed
    
    # Iterate each categorical column and create a subplot
    for i, column in enumerate(lst):
        plt.subplot(num_rows, num_cols, i + 1)  # create a subplot for each numeric
        ax = sns.histplot(data=data, x=column, kde=KDE, bins=bins)
        ax.grid(False)  # remove grid
        
        # customize each plot
        plt.title(f'Histogram Plot for {column} {txt}', fontsize=titleFont)
        plt.xlabel(column, fontsize=labelFont, fontweight='bold')
        plt.ylabel('Frequency', fontsize=labelFont, fontweight='bold')
        plt.xticks(fontsize=tickFont)
        plt.yticks(fontsize=tickFont)
    
    # adjust the space between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Show the plot
    plt.tight_layout()
    plt.show()


# # def histogramPlot(data, lst, bins=30, txt=''):
# #     # eet up the matplotlib figure
# #     num_cols = len(lst)
# #     plt.figure(figsize=(10, 5 * num_cols))  # adjust the figure size as needed
    
# #     # iterate each categorical column and create a subplot
# #     for i, column in enumerate(lst, start=1):
# #         plt.subplot(num_cols, 1, i)  # create a subplot for each numeric
# #         ax = sns.histplot(data=data, x=column, kde=True, bins=bins)
# #         ax.grid(False) # remove grid
        
# #         # customize each plot
# #         plt.title(f'Histogram Plot for {column} {txt}')
# #         plt.ylabel('Frequency')
    
# #     # adjust the vertical space between subplots
# #     plt.subplots_adjust(hspace=0.4)

# #     # show the plot
# #     plt.tight_layout()
# #     plt.show()


# def distributionByCategory(data, numeric, categorical, titleFont = 15, bins=30, title='Distribution of Numeric Feature by Category'):
#     """
#     Create a FacetGrid histogram for a numeric feature by a categorical feature.
#     """
#     # create the FacetGrid
#     g = sns.FacetGrid(data, col=categorical, height=4, aspect=1)
#     g.map(sns.histplot, numeric, bins=bins, kde=True)

#     # add a title
#     g.fig.suptitle(title, fontsize=titleFont)

#     # adjust the layout
#     g.fig.subplots_adjust(top=0.85)
#     plt.show()


# def categoricalFeaturesCountPlot(data, featureCat, txt='', figsize=(25, 15), tickFont=13, titleFont=12):
#     """
#     Create count plots for multiple categorical features.
#     """
#     # calculate the number of features and the grid dimensions
#     num_features = len(featureCat)
#     cols = 2  # set the number of columns
#     rows = (num_features + cols - 1) // cols  # Calculate the required rows

#     # set the figure size
#     plt.figure(figsize=figsize)

#     # create count plots for each categorical feature
#     for i, feature in enumerate(featureCat, start=1):
#         plt.subplot(rows, cols, i)  # Create subplot dynamically based on rows and columns
#         sns.countplot(data=data, y=feature, hue=feature, palette='Set2', legend=False)  # Use hue with the feature
#         plt.title(f'\n{feature} {txt}', fontsize=titleFont)
#         plt.xlabel("Frequency", fontsize=tickFont, fontweight='bold')
#         plt.ylabel(feature, fontsize=tickFont, fontweight='bold')
#         plt.xticks(fontsize=tickFont)
#         plt.yticks(fontsize=tickFont)
        
#     # adjust layout to prevent overlap
#     plt.tight_layout()
#     plt.show()


# def boxplotMultiple(df, numericCol, categoricalCol, hue, width=8, height=6):
#     """
#     Generate a boxplot of with one mumerical value.
#     """
#     # figure with the desired size
#     plt.figure(figsize=(width, height))
#     # create the boxplot
#     sns.boxplot(data=df, x=numericCol, y=categoricalCol, hue=hue)

#     # customize the legend
#     plt.legend(title=f"{hue}", loc='upper right', fontsize=10, frameon=True)

#     # add title
#     plt.title(f'Box Plot of {numericCol} by {categoricalCol} and {hue}\n')

#     # show the plot
#     plt.show()

# def safe_exp(x):
#     return np.exp(np.minimum(x, 709))  # 709 is roughly log(1e308), max float64 value


# def transformPlots(data, txt='', bins=30, figsize=(20, 5)):
#     """
#     Plots histograms of the original data, log-transformed data, 
#     square root-transformed data, and exponentially-transformed data 
#     in a single row of subplots.
#     """
#     # Transformations
#     log_data = np.log(data + 1)  # Adding 1 to avoid log(0)
#     sqrt_data = np.sqrt(data)
#     exp_data = safe_exp(data)  # Exponential transformation
    
#     # Creating subplots
#     fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    
#     # Plotting the histograms
#     axes[0].hist(data, bins=bins, color='blue', alpha=0.7, edgecolor='black')
#     axes[0].set_title(f"Original Data - ({txt})")
#     axes[0].set_xlabel(f"{txt}")
#     axes[0].set_ylabel("Frequency")
    
#     axes[1].hist(log_data, bins=bins, color='green', alpha=0.7, edgecolor='black')
#     axes[1].set_title(f"Log Transformed Data - ({txt})")
#     axes[1].set_xlabel(f"Log({txt} + 1)")
    
#     axes[2].hist(sqrt_data, bins=bins, color='orange', alpha=0.7, edgecolor='black')
#     axes[2].set_title(f"Square Root Transformed Data - ({txt})")
#     axes[2].set_xlabel(f"Sqrt({txt})")
    
#     axes[3].hist(exp_data, bins=bins, color='red', alpha=0.7, edgecolor='black')
#     axes[3].set_title(f"Exponential Transformed Data - ({txt})")
#     axes[3].set_xlabel(f"Exp({txt})")
    
#     # Adjust layout and show the plot
#     plt.tight_layout()
#     plt.show()


def DensityTransformPlots(data, txt='', bins=30, figsize=(15, 5), KDE=True):
    """
    Plots histograms of the original data, log-transformed data, 
    and square root-transformed data in a single row of subplots,
    with optional KDE overlays.
    """
    # Transformations
    log_data = np.log(data + 1)  # Adding 1 to avoid log(0)
    sqrt_data = np.sqrt(data)
    
    # Creating subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    
    # Plotting the histograms with optional KDE overlays
    for ax, transformed_data, title, xlabel, color in zip(
        axes,
        [data, log_data, sqrt_data],
        ["Original Data", "Log Transformed Data", "Square Root Transformed Data"],
        [f"{txt}", f"Log({txt} + 1)", f"Sqrt({txt})"],
        ['blue', 'green', 'orange']
    ):
        ax.hist(transformed_data, bins=bins, color=color, alpha=0.7, edgecolor='black', density=True)
        
        # Add KDE only if KDE=True
        if KDE:
            sns.kdeplot(transformed_data, ax=ax, color='red', lw=2)
        
        ax.set_title(f"{title} - ({txt})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()