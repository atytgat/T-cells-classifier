# T-cells classifier


The dataset consists of single-cell expression data, a source of data commonly used in cancer research, for example. To obtain it, a biopsy is performed: a surgeon extracts a small sample from the patient's tumor in order to analyze it. Each cell in the sample is then individually analyzed to obtain its gene expression profile. A single-cell dataset is thus composed of a collection of cells, corresponding to the rows, and the columns represent the different genes present in our DNA. Each value is interpreted as the expression level of a gene in a particular cell, i.e. the amount of proteins produced whenever that gene is read within the cell machinery.

The different cells composing a cell tissue generally present a high heterogeneity. There are a lot of different cell types, each performing a specific function,
and therefore presenting a different gene expression profile. In early days, medical researchers had no other option than to examine each cell by hand in
order determine their type. With the evolution of the sequencing technologies, datasets nowadays contain thousands, sometimes even millions of cells.
Anotating cells by hand is thus no longer an option. That's where machine learning techniques come in!

Indeed, the hand annotated datasets can be used to train machine learning algorithms in order to automatically classify new single-cell datasets. More
specifically, the task at hand is to distinguish between two kind of T-lymphocytes, which play an important role in our immune response against diseases, in this
case lung cancer. The dataset under study is composed of so called TCD4-lymphocytes. This category of T-cells is mainly composed of T helper cells, which
play a fundamental role in our adaptative immunity (.e. the immunity mechanisms that can adapt themselves to new diseases). However, a part of these
TCD4-lymphocytes will evolve into another category, called T regulatory cells. These cells will play an active role in preventing autoimmune diseases (i.e. when
our organism doesn't distinguish our cells from stranger cells and attacks itself). Since the evolution from T helper to T regulatory happens progressively, it
isn't always easy to distinguish between these two categories, making this task more challenging.


The data is composed of two parts:

/ML-A5-2020_ train. csv isa labeled dataset which contains 1000 rows representing the cells, named C-1 to C-1000. The last column, named label,
contains the class label of each cell: 1 for cells that evolved (or are evolving) to T regulatory cells, and -1 for the T helper cells.

¢ ML-A5-2020_ test .csv isa test set containing the 500 remaining cells, named C-1001 to C-1500. For this set, the class labels are not given, and it is your
task to make an as accurate prediction as possible of the type of each cell in this dataset.

The data matrix is composed of 23384 columns (+ 1 label column for the training set). Columns 1 to 23370 represent the different genes. The values in those
columns are the expression levels of the genes in each cell. Columns 23371 to 23384 contain additional information about the cells in the dataset, such as the
patient and the tissue they where taken from, their size, ete.



To tackle this problem, we used an ensemble learning approach to distinguish between T-helper and T-regulatory cells. Dimensionality reduction techniques are used to improve the generalization performances, and the SMOTE method was applied to balance the classes. More details on the methods used in the final model can be found in the Report file.

