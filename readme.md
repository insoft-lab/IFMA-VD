# IFMA-VD: Boosting Vulnerability Detection with Inter-Function Multilateral Association Insights

We proposed the IFMA-VD framework, which innovatively uses hypergraphs to model multilateral behavior associations. This framework includes an Inter-Function Multilateral Association analysis component designed to enhance vulnerability detection performance.

![image](https://github.com/user-attachments/assets/3cff7bec-4d86-4f45-b1f4-a61c174a40ac)

## The experimental results.

|      Datasets       | **FFmpeg** |        | **Qemu** |        | **Chrome+Debian** |        |
|:-------------------:|:----------:|:------:|:--------:|:------:|:-----------------:|:------:|
|     **Method**      |     F1     | Recall |    F1    | Recall |        F1         | Recall |
|  Devign(NIPS2019)   |    51.9    |  52.1  |   53.7   |  53.4  |       28.4        |  28.7  |
| CodeBERT(arXiv2020) |    53.0    |  51.2  |   54.1   |  52.3  |       25.4        |  27.4  |
|   IVDect(FSE2021)   |    65.5    |  64.6  |   57.9   |  64.6  |       38.8        |  39.5  |
|   Reveal(TSE2021)   |    62.6    |  82.4  |   49.3   |  54.0  |       26.3        |  28.6  |
|  VulCNN(ICSE2022)   |    54.2    |  57.7  |   55.1   |  58.2  |       31.5        |  51.0  |
|   VulBG(ICSE2023)   |    57.5    |  62.1  |   55.9   |  58.9  |       36.5        |  59.3  |
|  PDBert(ICSE2024)   |    52.4    |  43.2  |   62.4   |  59.8  |       47.9        |  45.4  |
|    IFMA-VD(ours)    |   =69.1=   | =95.9= |  =62.5=  | =74.4= |      =50.8=       | =78.6= |
|        W/T/L        |   7/0/0    | 7/0/0  |  7/0/0   | 7/0/0  |       7/0/0       | 7/0/0  |

The value enclosed by the '=' symbol represents the best in that column.


## Dataset

We conducted experiments using three widely recognized datasets in vulnerability detection research. The statistical data for these datasets are as follows:

|     Datasets     |  #Samples  |  #Vul   |  #Non-vul  |  Vul Ratio  |
|:----------------:|:----------:|:-------:|:----------:|:-----------:|
|      FFmpeg      |   9,768    |  4,981  |   4,788    |   51.10%    |
|       Qemu       |   17,549   |  7,479  |   10,070   |   42.62%    |
|  Chrome+Debian   |   22,734   |  2,240  |   20,494   |    9.85%    |

These three datasets are saved in `./data/`.

## Implementation

The framework includes three main stages:
1. **Parsing functions into code property graphs (CPG) and generating intra-function features.**
2. **Constructing a code behavior hypergraph and extracting inter-function multilateral association features.**
3. **Applying classification techniques to detect the presence of vulnerabilities.**

### Intra-Function Features Generation
- Parse data into CPG and PDG using [Joern](https://github.com/joernio/joern):
- Train intra-function feature extraction module from CPG.

### Inter-Function Features Generation
- Extract function behavior slices.
- Construct code behavior hypergraph.

### Vulnerability Detection
- Extract multilateral association features for vulnerability detection: `./IFMA-VD.py`

We provide pre-generated intra-function features and code behavior hypergraphs, available for download at [Processed dataset](https://drive.google.com/file/d/1e2QyEppFSOpOOWaXXFbTkIPYj3f4hDVK/view?usp=drive_link). After downloading, run `./IFMA-VD.py`.

## Checking Experimental Results

- Please check the folders `./results` for the experimental results.
