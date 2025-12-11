/**
 * @file rrwm_visual.cpp
 * @brief Implementation of Reweighted Random Walks for Graph Matching (RRWM) with Visualization.
 * 
 * This program reads an affinity matrix, reconstructs the graph topology, 
 * solves the graph matching problem using the RRWM algorithm, and visualizes 
 * the results using OpenCV.
 *
 * Reference: M. Cho, J. Lee, and K. M. Lee. "Reweighted Random Walks for Graph Matching". ECCV 2010.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ==========================================
// 1. Color Constants (OpenCV uses BGR format)
// ==========================================
const Scalar C_WHITE(255, 255, 255);
const Scalar C_BLACK(0, 0, 0);
const Scalar C_GRAY(200, 200, 200);
const Scalar C_ORANGE(60, 80, 255); // BGR: Orange (for matched/subgraph nodes)
const Scalar C_BLUE(180, 110, 30);  // BGR: Dark Blue (for unmatched global nodes)
const Scalar C_GREEN(0, 200, 0);    // BGR: Green (for matching connection lines)

// ==========================================
// 2. Data Reading and Topology Reconstruction
// ==========================================

/**
 * @brief Reads the Affinity Matrix from a text file.
 * 
 * @param filename Path to the text file containing the matrix data.
 * @param M Output parameter to store the dimension of the matrix (rows/cols).
 * @return cv::Mat The affinity matrix (square matrix of type CV_64F).
 */
Mat readAffinityMatrix(const string& filename, int& M) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(-1);
    }
    vector<double> values;
    double val;
    while (file >> val) values.push_back(val);

    // Validate that the matrix is square
    M = sqrt(values.size());
    if (M * M != values.size()) {
        cerr << "Error: Matrix is not square! Total elements: " << values.size() << endl;
        exit(-1);
    }

    Mat W(M, M, CV_64F);
    memcpy(W.data, values.data(), values.size() * sizeof(double));
    return W;
}

/**
 * @brief Reconstructs the adjacency matrices (topology) of the two graphs from the affinity matrix.
 * 
 * The affinity matrix K is constructed based on edge compatibilities.
 * A non-zero element at K(ia, jb) implies that the edge (i, j) exists in Graph 1
 * and the edge (a, b) exists in Graph 2.
 * 
 * @param K The affinity matrix.
 * @param n1 Number of nodes in Graph 1 (Global Graph).
 * @param n2 Number of nodes in Graph 2 (Subgraph).
 * @param adj1 Output adjacency matrix for Graph 1.
 * @param adj2 Output adjacency matrix for Graph 2.
 */
void reconstructTopology(const Mat& K, int n1, int n2, Mat& adj1, Mat& adj2) {
    // Initialize adjacency matrices with zeros
    adj1 = Mat::zeros(n1, n1, CV_8U);
    adj2 = Mat::zeros(n2, n2, CV_8U);

    // Iterate through the Affinity Matrix K
    // Row index 'r' corresponds to correspondence (u1, u2)
    // Col index 'c' corresponds to correspondence (v1, v2)
    for (int r = 0; r < K.rows; ++r) {
        for (int c = 0; c < K.cols; ++c) {
            // Check non-diagonal elements with a threshold to ignore noise
            if (r != c && K.at<double>(r, c) > 0.001) { 
                // Decode indices based on Kronecker product structure
                int u1 = r / n2; // Node in Graph 1
                int u2 = r % n2; // Node in Graph 2
                int v1 = c / n2; // Node in Graph 1
                int v2 = c % n2; // Node in Graph 2
                
                // If distinct nodes are connected in the assignment space, edges exist in original graphs
                if (u1 != v1) adj1.at<uchar>(u1, v1) = 1;
                if (u2 != v2) adj2.at<uchar>(u2, v2) = 1;
            }
        }
    }

    // Ensure the adjacency matrices are symmetric (Undirected Graphs)
    auto symmetrize = [](Mat& adj) {
        for(int i=0; i<adj.rows; ++i)
            for(int j=i+1; j<adj.cols; ++j)
                if(adj.at<uchar>(i,j) || adj.at<uchar>(j,i))
                    adj.at<uchar>(i,j) = adj.at<uchar>(j,i) = 1;
    };
    symmetrize(adj1);
    symmetrize(adj2);
}

// ==========================================
// 3. Graph Layout Algorithm
// ==========================================

/**
 * @brief Computes node positions using the Fruchterman-Reingold force-directed layout algorithm.
 * 
 * Nodes repel each other like charged particles, while connected edges act as springs 
 * pulling nodes together. This results in an aesthetically pleasing graph layout.
 * 
 * @param adj Adjacency matrix of the graph.
 * @param n Number of nodes.
 * @param positions Output vector of node coordinates.
 * @param area The size of the layout area.
 * @param seed Random seed for initialization stability.
 */
void computeSpringLayout(const Mat& adj, int n, vector<Point2f>& positions, Size area, int seed) {
    RNG rng(seed);
    positions.resize(n);
    
    // Padding to prevent nodes from sticking to the image borders
    float padding = 60.0f; 
    
    // Initialize with random positions
    for(int i=0; i<n; ++i) 
        positions[i] = Point2f(rng.uniform(area.width*0.2, area.width*0.8), 
                               rng.uniform(area.height*0.2, area.height*0.8));

    // Optimal distance between nodes
    float k = sqrt(area.width * area.height / (float)n) * 0.5; 
    float temp = area.width * 0.1; // Initial temperature
    
    // Iterative physics simulation
    for(int iter=0; iter<300; ++iter) { 
        vector<Point2f> disp(n, Point2f(0,0));
        
        // 1. Calculate Repulsive Forces (between all pairs of nodes)
        for(int v=0; v<n; ++v) {
            for(int u=0; u<n; ++u) {
                if(u==v) continue;
                Point2f delta = positions[v] - positions[u];
                float dist = norm(delta);
                if(dist < 15.0f) dist = 15.0f; // Prevent division by zero / extreme forces
                float force = k*k/dist;
                disp[v] += (delta/dist)*force;
            }
        }
        
        // 2. Calculate Attractive Forces (along edges)
        for(int v=0; v<n; ++v) {
            for(int u=0; u<n; ++u) {
                if(adj.at<uchar>(v,u)) {
                    Point2f delta = positions[v] - positions[u];
                    float dist = norm(delta);
                    if(dist < 15.0f) dist = 15.0f;
                    float force = dist*dist/k;
                    disp[v] -= (delta/dist)*force;
                }
            }
        }
        
        // 3. Update Positions and Cool Down
        for(int v=0; v<n; ++v) {
            float d = norm(disp[v]);
            if(d>0) {
                // Limit the maximum displacement by current temperature
                positions[v] += (disp[v]/d) * min(d, temp);
                
                // Constrain nodes within the bounding box
                positions[v].x = min((float)area.width-padding, max(padding, positions[v].x));
                positions[v].y = min((float)area.height-padding, max(padding, positions[v].y));
            }
        }
        temp *= 0.95; // Cooling schedule
    }
}

// ==========================================
// 4. RRWM Algorithm Implementation
// ==========================================

/**
 * @brief Performs Sinkhorn normalization (Doubly Stochastic Normalization).
 * 
 * Iteratively normalizes rows and columns to ensure they sum to 1.
 * This enforces the one-to-one matching constraint in the continuous domain.
 * 
 * @param x_vec The vectorized assignment matrix.
 * @param rows Number of rows (Graph 1 nodes).
 * @param cols Number of columns (Graph 2 nodes).
 * @return cv::Mat The normalized vector.
 */
Mat sinkhorn(const Mat& x_vec, int rows, int cols) {
    Mat Y = x_vec.reshape(1, rows).clone();
    // Usually converges quickly, 10 iterations are sufficient
    for(int k=0; k<10; ++k) {
        // Row Normalization
        for(int i=0; i<rows; ++i) { 
            double s=0; for(int j=0; j<cols; ++j) s+=Y.at<double>(i,j);
            if(s>1e-9) Y.row(i) /= s;
        }
        // Column Normalization
        for(int j=0; j<cols; ++j) { 
            double s=0; for(int i=0; i<rows; ++i) s+=Y.at<double>(i,j);
            if(s>1e-9) Y.col(j) /= s;
        }
    }
    return Y.reshape(1, rows*cols);
}

/**
 * @brief The core RRWM solver.
 * 
 * Solves the Graph Matching problem x* = argmax(x^T * W * x) subject to mapping constraints.
 * 
 * @param W The affinity matrix.
 * @param n1 Number of nodes in Graph 1.
 * @param n2 Number of nodes in Graph 2.
 * @return cv::Mat Binary assignment matrix (n1 x n2) indicating matches.
 */
Mat RRWM(const Mat& W, int n1, int n2) {
    // Hyperparameters as per the paper
    double alpha = 0.2; // Trade-off between random walk and reweighting
    double beta = 30.0; // Inflation parameter for exponential reweighting
    
    // Normalize W to create a transition matrix P
    Mat ones = Mat::ones(W.cols, 1, CV_64F);
    Mat col_sums = W * ones;
    double d_max; minMaxLoc(col_sums, NULL, &d_max);
    Mat P = W / (d_max + 1e-10); // Avoid division by zero
    
    // Initialize state vector x uniformly
    Mat x = Mat::ones(n1*n2, 1, CV_64F) / (n1*n2);
    
    // --- Main Iteration Loop ---
    for(int i=0; i<50; ++i) {
        Mat x_old = x.clone();
        
        // 1. Random Walk Step
        Mat bar_x = P.t() * x;
        
        // 2. Reweighting Step (Inflation)
        double mx; minMaxLoc(bar_x, NULL, &mx);
        Mat y; exp(beta * bar_x / (mx+1e-10), y);
        
        // 3. Bistochastic Normalization (Sinkhorn)
        y = sinkhorn(y, n1, n2);
        y /= (sum(y)[0] + 1e-10); // Global normalization
        
        // 4. Update Step (Combine original walk and reweighted result)
        x = alpha*bar_x + (1-alpha)*y;
        x /= (sum(x)[0] + 1e-10);
        
        // Check convergence
        if(norm(x-x_old) < 1e-5) break;
    }
    
    // --- Discretization (Greedy Approach) ---
    // Convert continuous scores to binary assignment
    Mat X_res = Mat::zeros(n1, n2, CV_8U);
    Mat X_temp = x.reshape(1, n1).clone();
    
    // We expect at most 'n2' matches (since n2 < n1 in this subgraph case)
    for(int k=0; k<n2; ++k) { 
        double maxVal; Point maxLoc;
        minMaxLoc(X_temp, NULL, &maxVal, NULL, &maxLoc);
        
        if(maxVal < 1e-6) break; // No more valid matches
        
        X_res.at<uchar>(maxLoc.y, maxLoc.x) = 1;
        
        // Suppress the selected row and column to enforce one-to-one constraint
        X_temp.row(maxLoc.y).setTo(-1);
        X_temp.col(maxLoc.x).setTo(-1);
    }
    return X_res;
}

// ==========================================
// 5. Large-Scale Visualization
// ==========================================

/**
 * @brief visualizes the matching result on a large canvas.
 * 
 * Draws the Subgraph and Global Graph in separate boxes, rendering internal edges
 * and connections indicating the matches found by the algorithm.
 * 
 * @param X Binary matching matrix.
 * @param adj_sub Adjacency matrix of the subgraph.
 * @param pos_sub Node positions of the subgraph.
 * @param col_sub Node colors of the subgraph.
 * @param adj_global Adjacency matrix of the global graph.
 * @param pos_global Node positions of the global graph.
 * @param col_global Node colors of the global graph.
 */
void visualizeFinal(
    const Mat& X, 
    const Mat& adj_sub, const vector<Point2f>& pos_sub, const vector<Scalar>& col_sub,
    const Mat& adj_global, const vector<Point2f>& pos_global, const vector<Scalar>& col_global
) {
    // Set up a large high-resolution canvas (2000x1200)
    int CanvasW = 2000;
    int CanvasH = 1200;
    Mat canvas(CanvasH, CanvasW, CV_8UC3, C_WHITE);
    
    // Layout parameters for the bounding boxes
    int BoxSize = 900;       // Size of the square region for each graph
    int MarginX = 60;        // Left margin
    int MarginY = 150;       // Top margin (for titles)
    int Gap = 80;            // Gap between the two graph boxes
    
    // Define regions of interest (ROI)
    Rect r_sub(MarginX, MarginY, BoxSize, BoxSize);     
    Rect r_global(MarginX + BoxSize + Gap, MarginY, BoxSize, BoxSize); 
    
    Point2f off_sub(r_sub.x, r_sub.y);
    Point2f off_global(r_global.x, r_global.y);

    // 1. Draw Bounding Boxes and Titles
    rectangle(canvas, r_sub, C_BLACK, 3);    
    rectangle(canvas, r_global, C_BLACK, 3); 
    
    // Use large fonts for headers
    putText(canvas, "Subgraph (5 nodes)", Point(r_sub.x + 250, r_sub.y - 40), 
            FONT_HERSHEY_SIMPLEX, 2.0, C_BLACK, 3);
    putText(canvas, "Global Graph (10 nodes)", Point(r_global.x + 200, r_global.y - 40), 
            FONT_HERSHEY_SIMPLEX, 2.0, C_BLACK, 3);

    // 2. Draw Internal Graph Edges (Black, Thick)
    int edge_thickness = 3;
    
    // Draw edges for Subgraph
    for(int i=0; i<adj_sub.rows; ++i) {
        for(int j=i+1; j<adj_sub.cols; ++j) {
            if(adj_sub.at<uchar>(i,j))
                line(canvas, pos_sub[i]+off_sub, pos_sub[j]+off_sub, C_BLACK, edge_thickness, LINE_AA);
        }
    }
    // Draw edges for Global Graph
    for(int i=0; i<adj_global.rows; ++i) {
        for(int j=i+1; j<adj_global.cols; ++j) {
            if(adj_global.at<uchar>(i,j))
                line(canvas, pos_global[i]+off_global, pos_global[j]+off_global, C_BLACK, edge_thickness, LINE_AA);
        }
    }

    // 3. Draw Matching Connections (Green, Thick)
    // These lines connect nodes from the Subgraph to the Global Graph based on the result X
    int match_line_thickness = 3;
    for(int i=0; i<X.rows; ++i) { // i: Global node index
        for(int j=0; j<X.cols; ++j) { // j: Sub node index
            if(X.at<uchar>(i, j) == 1) {
                Point2f p_sub = pos_sub[j] + off_sub;
                Point2f p_global = pos_global[i] + off_global;
                line(canvas, p_sub, p_global, C_GREEN, match_line_thickness, LINE_AA);
            }
        }
    }

    // 4. Draw Nodes (Large circles with IDs)
    // Drawn last to cover line endpoints for a cleaner look
    int node_radius = 30; 
    double font_scale = 1.0; 
    int font_thick = 2;
    
    auto drawNodes = [&](const vector<Point2f>& pts, Point2f off, const vector<Scalar>& colors) {
        for(int i=0; i<pts.size(); ++i) {
            Point2f p = pts[i] + off;
            // Draw filled circle
            circle(canvas, p, node_radius, colors[i], -1, LINE_AA); 
            // Draw black outline for contrast
            circle(canvas, p, node_radius, C_BLACK, 2, LINE_AA);
            
            // Draw Node ID centered
            string id = to_string(i);
            int base;
            Size sz = getTextSize(id, FONT_HERSHEY_SIMPLEX, font_scale, font_thick, &base);
            putText(canvas, id, p + Point2f(-sz.width/2, sz.height/2), 
                    FONT_HERSHEY_SIMPLEX, font_scale, C_WHITE, font_thick, LINE_AA);
        }
    };

    drawNodes(pos_sub, off_sub, col_sub);
    drawNodes(pos_global, off_global, col_global);

    // 5. Window Management
    // Use WINDOW_NORMAL to allow resizing the window manually if the screen resolution is low
    namedWindow("RRWM Large Visualization", WINDOW_NORMAL);
    imshow("RRWM Large Visualization", canvas);
    resizeWindow("RRWM Large Visualization", 1500, 900); // Set a reasonable initial viewing size
    waitKey(0);
}

int main() {
    string file = "../K_matrix.txt";
    int M;
    
    // 1. Read the Affinity Matrix
    Mat K = readAffinityMatrix(file, M); // Dimension M usually equals n_global * n_sub
    
    // Define Graph Sizes based on problem constraints
    int n_global = 10;
    int n_sub = 5;
    
    // 2. Reconstruct Graph Topology from K
    Mat adj_global, adj_sub;
    reconstructTopology(K, n_global, n_sub, adj_global, adj_sub);
    
    // 3. Solve the Graph Matching Problem
    Mat X = RRWM(K, n_global, n_sub);
    
    // 4. Compute Layout for Visualization
    // Use a large layout area (900x900) for better separation of nodes
    Size layoutSize(900, 900); 
    vector<Point2f> pos_sub, pos_global;
    
    // Compute force-directed layout
    computeSpringLayout(adj_sub, n_sub, pos_sub, layoutSize, 100);
    computeSpringLayout(adj_global, n_global, pos_global, layoutSize, 200);
    
    // 5. Determine Node Colors
    vector<Scalar> col_sub(n_sub, C_ORANGE);   // Subgraph nodes are always Orange
    vector<Scalar> col_global(n_global, C_BLUE); // Default Global nodes are Blue
    
    // Highlight matched nodes in Global Graph with Orange
    for(int i=0; i<n_global; ++i) {
        for(int j=0; j<n_sub; ++j) {
            if(X.at<uchar>(i, j) == 1) {
                col_global[i] = C_ORANGE; 
            }
        }
    }

    // 6. Final Visualization
    visualizeFinal(X, adj_sub, pos_sub, col_sub, adj_global, pos_global, col_global);
    
    return 0;
}
