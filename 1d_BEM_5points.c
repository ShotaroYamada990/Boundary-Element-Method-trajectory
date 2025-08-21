#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Constants
#define N 5  // Number of nodes
static const double gauss_points[2] = {-1.0/sqrt(3.0), 1.0/sqrt(3.0)};
static const double gauss_weights[2] = {1.0, 1.0};
static const double epsilon0 = 8.854e-12;
static const double tol = 1e-10;
static const double delta = 1e-6;

// Define nodes array
double nodes[N] = {-1.0, -0.5, 0.0, 0.5, 1.0};

// Basis function for node j
double phai_j(double x_prime, int j) {
    if (j == 0) {
        return (x_prime <= nodes[1]) ? (nodes[1] - x_prime)/(nodes[1]-nodes[0]) : 0.0;
    }
    else if (j == N-1) {
        return (x_prime >= nodes[N-2]) ? (x_prime - nodes[N-2])/(nodes[N-1]-nodes[N-2]) : 0.0;
    }
    else {
        if (x_prime >= nodes[j-1] && x_prime <= nodes[j]) {
            return (x_prime - nodes[j-1])/(nodes[j]-nodes[j-1]);
        }
        else if (x_prime >= nodes[j] && x_prime <= nodes[j+1]) {
            return (nodes[j+1] - x_prime)/(nodes[j+1]-nodes[j]);
        }
        return 0.0;
    }
}

// Compute segment integral with singularity handling
double compute_segment_integral(double x_i, double a, double b, int j, int role) {
    double integral = 0.0;
    double singular_term = 0.0;
    double segment_length = b - a;

    // Check if x_i is in this segment
    if (x_i >= a - tol && x_i <= b + tol) {
        // Left endpoint singularity
        if (fabs(x_i - a) < tol) {
            if (role == 0) { // j is left node
                singular_term = delta * log(delta) - delta;
                // Integrate from a+delta to b
                segment_length = b - (a + delta);
                for (int k = 0; k < 2; k++) {
                    double xi = gauss_points[k];
                    double x_prime = (a + delta) + segment_length * (xi + 1.0)/2.0;
                    double r = fabs(x_i - x_prime);
                    double basis_val = phai_j(x_prime, j);
                    integral += gauss_weights[k] * basis_val * log(r) * segment_length / 2.0;
                }
            }
            else { // j is right node - no singularity
                for (int k = 0; k < 2; k++) {
                    double xi = gauss_points[k];
                    double x_prime = a + segment_length * (xi + 1.0)/2.0;
                    double r = fabs(x_i - x_prime);
                    double basis_val = phai_j(x_prime, j);
                    integral += gauss_weights[k] * basis_val * log(r) * segment_length / 2.0;
                }
            }
        }
        // Right endpoint singularity
        else if (fabs(x_i - b) < tol) {
            if (role == 1) { // j is right node
                singular_term = delta * log(delta) - delta;
                // Integrate from a to b-delta
                segment_length = (b - delta) - a;
                for (int k = 0; k < 2; k++) {
                    double xi = gauss_points[k];
                    double x_prime = a + segment_length * (xi + 1.0)/2.0;
                    double r = fabs(x_i - x_prime);
                    double basis_val = phai_j(x_prime, j);
                    integral += gauss_weights[k] * basis_val * log(r) * segment_length / 2.0;
                }
            }
            else { // j is left node - no singularity
                for (int k = 0; k < 2; k++) {
                    double xi = gauss_points[k];
                    double x_prime = a + segment_length * (xi + 1.0)/2.0;
                    double r = fabs(x_i - x_prime);
                    double basis_val = phai_j(x_prime, j);
                    integral += gauss_weights[k] * basis_val * log(r) * segment_length / 2.0;
                }
            }
        }
        // Interior point
        else {
            for (int k = 0; k < 2; k++) {
                double xi = gauss_points[k];
                double x_prime = a + segment_length * (xi + 1.0)/2.0;
                double r = fabs(x_i - x_prime);
                double basis_val = phai_j(x_prime, j);
                integral += gauss_weights[k] * basis_val * log(r) * segment_length / 2.0;
            }
        }
    }
    // Non-singular case
    else {
        for (int k = 0; k < 2; k++) {
            double xi = gauss_points[k];
            double x_prime = a + segment_length * (xi + 1.0)/2.0;
            double r = fabs(x_i - x_prime);
            double basis_val = phai_j(x_prime, j);
            integral += gauss_weights[k] * basis_val * log(r) * segment_length / 2.0;
        }
    }
    return singular_term + integral;
}

// Compute potential matrix element A[i][j]
double compute_Aij(double x_i, int j) {
    double total_integral = 0.0;

    // Left segment (if exists)
    if (j > 0) {
        double a = nodes[j-1];
        double b = nodes[j];
        total_integral += compute_segment_integral(x_i, a, b, j, 1); // j is right node
    }

    // Right segment (if exists)
    if (j < N-1) {
        double a = nodes[j];
        double b = nodes[j+1];
        total_integral += compute_segment_integral(x_i, a, b, j, 0); // j is left node
    }

    return -total_integral / (2.0 * M_PI * epsilon0);
}

// LU Decomposition Solver for NxN systems
void solve_system(int n, double A[][N], double b[], double x[]) {
    double L[N][N], U[N][N];
    double y[N];

    // Initialize matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }

    // LU decomposition
    for (int k = 0; k < n; k++) {
        U[k][k] = A[k][k];
        for (int i = k+1; i < n; i++) {
            L[i][k] = A[i][k] / U[k][k];
            U[k][i] = A[k][i];
        }
        for (int i = k+1; i < n; i++) {
            for (int j = k+1; j < n; j++) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    // Forward substitution (Ly = b)
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }

    // Backward substitution (Ux = y)
    for (int i = n-1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i+1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
}

// Compute potential at any point x using solved charge densities
double compute_V_at_point(double x, double sigma[]) {
    double V = 0.0;
    for (int j = 0; j < N; j++) {
        V += compute_Aij(x, j) * sigma[j];
    }
    return V;
}

int main() {
    double A[N][N];              // BEM matrix
    double V_boundary[N];         // Boundary potential vector
    double sigma[N];              // Charge density solution

    // Set boundary conditions (constant 1V potential)
    for (int i = 0; i < N; i++) {
        V_boundary[i] = 1.0;
    }

    // Build matrix A: potential at node i due to basis j
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = compute_Aij(nodes[i], j);
        }
    }

    // Solve the system: A * sigma = V_boundary
    solve_system(N, A, V_boundary, sigma);

    // Print solution
    printf("Solved charge densities:\n");
    for (int i = 0; i < N; i++) {
        printf("σ[%d] = %e C/m\n", i, sigma[i]);
    }

    // Potential verification
    double test_points[] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    int num_points = sizeof(test_points) / sizeof(test_points[0]);
    
    printf("\nPotential verification throughout the electrode:\n");
    printf("%-10s %-15s %-15s\n", "x", "Computed V", "Error");
    for (int i = 0; i < num_points; i++) {
        double x = test_points[i];
        double V = compute_V_at_point(x, sigma);
        double error = fabs(V - 1.0);
        printf("%-10.1f %-15.6e %-15.6e\n", x, V, error);
    }

    /*
    // Particle simulation setup (commented out for now)
    init_normalization();  // Computes c = sqrt(m/fabs(q))
    double x0 = 0.0;       // Initial x-position [m]
    double y0 = 1.0;       // Initial y-position [m]
    double vx0_phys = 1.0; // Initial x-velocity [m/s]
    double vy0_phys = 0.0; // Initial y-velocity [m/s]
    const double ds = 0.1;
    const double dt_physical = ds * c; // Time step [s]
    int steps = 1000;      // Number of steps

    simulate_trajectory_normalized(x0, y0, vx0_phys, vy0_phys, sigma, dt_physical, steps);
    */
    
    return 0;
}