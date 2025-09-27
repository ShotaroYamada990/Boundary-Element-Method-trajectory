/*
Axisymmetric Electrostatics BEM (single-layer, Dirichlet BC)

- Unknown: lambda(s) = 2*pi*R(s)*sigma(s), linear surface charge along the generator (C/m).
- Potential kernel for a ring (per unit total ring charge):
    G(r,z; R,Z) = (1 / (2 * pi^2 * eps0)) * K(k) / d
  where:
    d^2 = (r + R)^2 + (z - Z)^2
    k^2 = 4 * r * R / d^2
    K(k) = complete elliptic integral of the first kind (modulus k).
- Single-layer representation on the surface gives:
    V(x_i) = ∫ G(x_i; y(s)) * lambda(s) ds
  which we discretize with linear elements along the generator curve in the R–Z plane.

This program:
- Builds a sphere of radius a as a surface of revolution (generator: semicircle).
- Assembles the dense matrix A (collocation at nodes, linear shape functions on elements).
- Solves A * lambda = V, with V=1 volt on the conductor.
- Reports lambda(s) and the total charge Q; compares to exact sphere Q_exact = 4*pi*eps0*a.

Compile:
  gcc -O2 -std=c11 axisym_bem.c -lm

Notes:
- No external libraries required; elliptic integrals computed by AGM.
- For clarity and robustness, a 32-point Gauss–Legendre rule is used; near-singular self terms use simple sub-segmentation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

/* Physical constant (SI). You can set eps0=1.0 for dimensionless tests. */
static const double eps0 = 8.854187817e-12;
const double V0 = 1.0; 
/* Gauss–Legendre nodes/weights for n=32 (positive abscissas only). */
static const int GLN = 32;
static const int GLH = 16; /* half (positive side) */
static const double gl_x[16] = {
  0.0483076656877383162348126,
  0.1444719615827964934851864,
  0.2392873622521370745446032,
  0.3318686022821276497799168,
  0.4213512761306353453641194,
  0.5068999089322293900237475,
  0.5877157572407623290407455,
  0.6630442669302152009751152,
  0.7321821187402896803874267,
  0.7944837959679424069630973,
  0.8493676137325699701336930,
  0.8963211557660521239653072,
  0.9349060759377396891709191,
  0.9647622555875064307738119,
  0.9856115115452683354001750,
  0.9972638618494815635449811
};
static const double gl_w[16] = {
  0.0965400885147278005667648,
  0.0956387200792748594190820,
  0.0938443990808045656391802,
  0.0911738786957638847128686,
  0.0876520930044038111427715,
  0.0833119242269467552221991,
  0.0781938957870703064717409,
  0.0723457941088485062253994,
  0.0658222227763618468376501,
  0.0586840934785355471452836,
  0.0509980592623761761961632,
  0.0428358980222266806568786,
  0.0342738629130214331026877,
  0.0253920653092620594557526,
  0.0162743947309056706051706,
  0.0070186100094700966004071
};

/* Data structures for generator curve */
typedef struct { double R, Z; } Node;
typedef struct { int n1, n2; double length; } Elem;

/* Safe utilities */
static void* xcalloc(size_t n, size_t sz) {
  void* p = calloc(n, sz);
  if (!p) { fprintf(stderr, "Out of memory\n"); exit(1); }
  return p;
}

static double clamp(double x, double lo, double hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/* Complete elliptic integrals K(k), E(k) via AGM; k in [0,1). */
static void ellip_KE(double k, double* K, double* E) {
  k = fabs(k);
  if (k >= 1.0) {
    /* Approach k -> 1-: K diverges, E -> 1. Handle softly. */
    double eps = 1e-15;
    double kk = 1.0 - eps;
    ellip_KE(kk, K, E);
    return;
  }
  if (k == 0.0) {
    *K = M_PI / 2.0;
    *E = M_PI / 2.0;
    return;
  }
  /* AGM iteration */
  double a = 1.0;
  double b = sqrt(1.0 - k*k);
  double c = 0.0;
  double sum = 0.0;
  double pow2 = 1.0;
  const int maxit = 50;
  const double tol = 1e-16;
  for (int n = 0; n < maxit; ++n) {
    c = 0.5 * (a - b);
    double an = 0.5 * (a + b);
    double bn = sqrt(a * b);
    sum += pow2 * c * c;
    pow2 *= 2.0;
    a = an;
    b = bn;
    if (fabs(c) < tol * a) break;
  }
  *K = M_PI / (2.0 * a);
  *E = (*K) * (1.0 - sum);
}

/* Axisymmetric ring kernel (per unit total ring charge) factor:
   G = (1/(2*pi^2*eps0)) * K(k) / d, with:
     d = sqrt((r + R)^2 + (z - Z)^2), k^2 = 4*r*R / d^2
*/
static double ring_kernel(double r, double z, double R, double Z) {
  /* Handle exact coincidence (rare in practice due to measure zero on surface). */
  double d2 = (r + R)*(r + R) + (z - Z)*(z - Z);
  double d = sqrt(d2);
  if (d == 0.0) {
    return 0.0; /* harmless fallback; integral stays finite */
  }
  double k2 = 0.0;
  if (r > 0.0 && R > 0.0) {
    k2 = clamp(4.0 * r * R / d2, 0.0, 1.0 - 1e-16);
  }
  double k = sqrt(k2);
  double Kc, Ec;
  ellip_KE(k, &Kc, &Ec);
  const double c0 = 1.0 / (2.0 * M_PI * M_PI * eps0);
  return c0 * (Kc / d);
}

/* Map local coordinate s in [-1,1] to a point on element (linear) */
static void map_linear(const Node* a, const Node* b, double s, double* R, double* Z) {
  /* Linear blending: p(s) = 0.5*(1-s)*a + 0.5*(1+s)*b */
  double Na = 0.5 * (1.0 - s);
  double Nb = 0.5 * (1.0 + s);
  *R = Na * a->R + Nb * b->R;
  *Z = Na * a->Z + Nb * b->Z;
}

/* Precompute element lengths */
static void compute_elem_lengths(Elem* elems, const Node* nodes, int ne) {
  for (int e = 0; e < ne; ++e) {
    Node a = nodes[elems[e].n1];
    Node b = nodes[elems[e].n2];
    double dR = b.R - a.R;
    double dZ = b.Z - a.Z;
    elems[e].length = sqrt(dR*dR + dZ*dZ);
  }
}

/* Assemble dense matrix A (N x N). Linear elements, node-collocation. */
static void assemble_A(double* A, const Node* nodes, const Elem* elems, int nn, int ne) {
  /* Zero A */
  for (int i = 0; i < nn*nn; ++i) A[i] = 0.0;

  for (int i = 0; i < nn; ++i) {
    const double ri = nodes[i].R;
    const double zi = nodes[i].Z;

    for (int e = 0; e < ne; ++e) {
      int j1 = elems[e].n1;
      int j2 = elems[e].n2;
      Node a = nodes[j1], b = nodes[j2];

      /* Linear shape functions on element: N1(s)=(1-s)/2, N2(s)=(1+s)/2 */
      double Le = elems[e].length;
      double J = 0.5 * Le;

      /* Near-singular boost: subdivide if collocation node touches the element */
      int subdiv = (i == j1 || i == j2) ? 8 : 1;

      for (int sub = 0; sub < subdiv; ++sub) {
        double s0 = -1.0 + 2.0 * (double)sub / (double)subdiv;
        double s1 = -1.0 + 2.0 * (double)(sub + 1) / (double)subdiv;
        double sm = 0.5 * (s0 + s1);
        double half = 0.5 * (s1 - s0);

        double contrib_j1 = 0.0;
        double contrib_j2 = 0.0;

        /* 32-point Gauss–Legendre over this subinterval */
        for (int k = 0; k < GLH; ++k) {
          double x = gl_x[k];
          double w = gl_w[k];

          /* Evaluate at +/- nodes within subinterval */
          for (int sign = -1; sign <= 1; sign += 2) {
            double s = sm + sign * half * x;
            double Rq, Zq;
            map_linear(&a, &b, s, &Rq, &Zq);

            double G = ring_kernel(ri, zi, Rq, Zq);
            double N1 = 0.5 * (1.0 - s);
            double N2 = 0.5 * (1.0 + s);

            double weight = w * half * J;
            contrib_j1 += G * N1 * weight;
            contrib_j2 += G * N2 * weight;
          }
        }

        A[i*nn + j1] += contrib_j1;
        A[i*nn + j2] += contrib_j2;
      }
    }
  }
}

/* Solve A x = b using Gaussian elimination with partial pivoting (in-place). */
static int solve_linear(int n, double* A, double* b, double* x) {
  /* Copy b to x (will be overwritten) */
  for (int i = 0; i < n; ++i) x[i] = b[i];

  /* LU factorization with partial pivoting (naive implementation) */
  for (int k = 0; k < n; ++k) {
    /* Pivot */
    int piv = k;
    double amax = fabs(A[k*n + k]);
    for (int i = k+1; i < n; ++i) {
      double v = fabs(A[i*n + k]);
      if (v > amax) { amax = v; piv = i; }
    }
    if (amax < 1e-30) return -1; /* singular */

    if (piv != k) {
      /* Swap rows piv <-> k in A and x */
      for (int j = k; j < n; ++j) {
        double tmp = A[k*n + j];
        A[k*n + j] = A[piv*n + j];
        A[piv*n + j] = tmp;
      }
      double tmpb = x[k];
      x[k] = x[piv];
      x[piv] = tmpb;
    }

    /* Eliminate */
    for (int i = k+1; i < n; ++i) {
      double f = A[i*n + k] / A[k*n + k];
      A[i*n + k] = 0.0;
      for (int j = k+1; j < n; ++j) {
        A[i*n + j] -= f * A[k*n + j];
      }
      x[i] -= f * x[k];
    }
  }

  /* Back substitution */
  for (int i = n-1; i >= 0; --i) {
    double sum = x[i];
    for (int j = i+1; j < n; ++j) sum -= A[i*n + j] * x[j];
    x[i] = sum / A[i*n + i];
  }
  return 0;
}

/* Build generator for a sphere of radius a using N nodes (theta from 0..pi). */
static void build_sphere(double a, int N, Node* nodes, Elem* elems) {
  /* Nodes */
  for (int i = 0; i < N; ++i) {
    double t = (double)i / (double)(N - 1);
    double th = M_PI * t;       /* 0 .. pi */
    nodes[i].R = a * sin(th);
    nodes[i].Z = a * cos(th);
  }
  /* Elements */
  for (int e = 0; e < N - 1; ++e) {
    elems[e].n1 = e;
    elems[e].n2 = e + 1;
  }
  compute_elem_lengths(elems, nodes, N - 1);
}

/* Integrate total charge Q = ∫ lambda ds using element-wise trapezoid rule. */
static double total_charge(const Node* nodes, const Elem* elems, const double* lambda, int ne) {
  double Q = 0.0;
  for (int e = 0; e < ne; ++e) {
    int i = elems[e].n1, j = elems[e].n2;
    double lam_avg = 0.5 * (lambda[i] + lambda[j]);
    Q += lam_avg * elems[e].length;
  }
  return Q;
}

/* Evaluate potential at a field point (r,z) from solved lambda(s). */
static double evaluate_potential(double r, double z, const Node* nodes, const Elem* elems, int nn, int ne, const double* lambda) {
  double V = 0.0;
  for (int e = 0; e < ne; ++e) {
    int j1 = elems[e].n1, j2 = elems[e].n2;
    Node a = nodes[j1], b = nodes[j2];
    double Le = elems[e].length;
    double J = 0.5 * Le;

    double val = 0.0;
    for (int k = 0; k < GLH; ++k) {
      double x = gl_x[k], w = gl_w[k];
      for (int sign = -1; sign <= 1; sign += 2) {
        double s = sign * x;
        double Rq, Zq;
        map_linear(&a, &b, s, &Rq, &Zq);
        double N1 = 0.5 * (1.0 - s);
        double N2 = 0.5 * (1.0 + s);
        double lam = N1 * lambda[j1] + N2 * lambda[j2];
        double G = ring_kernel(r, z, Rq, Zq);
        val += w * G * lam * J;
      }
    }
    V += val;
  }
  return V;
}

static void ring_field_kernel(double r, double z, double R, double Z, double *Er, double *Ez) {
    const double tiny = 1e-20;
    double A = (r + R)*(r + R) + (z - Z)*(z - Z); // d^2
    if (A < tiny) { *Er = 0.0; *Ez = 0.0; return; }
    double d = sqrt(A);

    double k2 = 4.0 * r * R / A;
    if (k2 < 0.0) k2 = 0.0;
    if (k2 >= 1.0) k2 = 1.0 - 1e-15;
    double k = sqrt(k2);

    double Kk, Ek;
    ellip_KE(k, &Kk, &Ek); /* AGM routine you already have */

    double dA_dr = 2.0 * (r + R);
    double dA_dz = 2.0 * (z - Z);
    double dk2_dr = 4.0 * R / A - 4.0 * r * R * dA_dr / (A * A);
    double dk2_dz =      - 4.0 * r * R * dA_dz / (A * A);

    double dk_dr = (k > 0.0) ? 0.5 * dk2_dr / k : 0.0;
    double dk_dz = (k > 0.0) ? 0.5 * dk2_dz / k : 0.0;

    double dd_dr = (r + R) / d;
    double dd_dz = (z - Z) / d;

    double dK_dk;
    if (k < 1e-12) {
        dK_dk = 0.0; /* small-k approx; improve if needed */
    } else {
        dK_dk = (Ek - (1.0 - k*k) * Kk) / (k * (1.0 - k*k));
    }

    double pref = 1.0 / (2.0 * M_PI * M_PI * eps0);

    double dG_dr = pref * ( (1.0 / d) * dK_dk * dk_dr - (Kk / (d * d)) * dd_dr );
    double dG_dz = pref * ( (1.0 / d) * dK_dk * dk_dz - (Kk / (d * d)) * dd_dz );

    /* return electric field produced by unit line-charge: Er = -dG/dr, Ez = -dG/dz */
    *Er = -dG_dr;
    *Ez = -dG_dz;
}

static void evaluate_electric_field(double r, double z, const Node* nodes, const Elem* elems,
                                    int nn, int ne, const double* lambda, double *Er, double *Ez) {
    double Er_total = 0.0;
    double Ez_total = 0.0;

    for (int e = 0; e < ne; ++e) {
        int j1 = elems[e].n1;
        int j2 = elems[e].n2;
        Node a = nodes[j1], b = nodes[j2];
        double Le = elems[e].length;
        double J = 0.5 * Le; /* Jacobian from reference [-1,1] to element */

        /* subdiv logic if you use (subdiv>1) for near-singular handling */
        int subdiv = 1;
        for (int sub = 0; sub < subdiv; ++sub) {
            double s0 = -1.0 + 2.0*sub / (double)subdiv;
            double s1 = -1.0 + 2.0*(sub+1) / (double)subdiv;
            double sm = 0.5*(s0 + s1);
            double half = 0.5*(s1 - s0);

            for (int k = 0; k < GLH; ++k) {
                double x = gl_x[k], w = gl_w[k];
                for (int sign = -1; sign <= 1; sign += 2) {
                    double s = sm + half * (sign * x);
                    double N1 = 0.5 * (1.0 - s);
                    double N2 = 0.5 * (1.0 + s);

                    double Rq, Zq;
                    map_linear(&a, &b, s, &Rq, &Zq); /* you already have this */

                    double lamq = N1 * lambda[j1] + N2 * lambda[j2];

                    double dEr, dEz;
                    ring_field_kernel(r, z, Rq, Zq, &dEr, &dEz);

                    double weight = w * half * J; /* same as in potential assembly */
                    Er_total += lamq * dEr * weight;
                    Ez_total += lamq * dEz * weight;
                }
            }
        }
    }

    *Er = Er_total;
    *Ez = Ez_total;
}


int main(void) {
  /* Geometry parameters */
  const double a = 1.0;        /* sphere radius (meters) */
  const int N = 160;           /* number of nodes along generator */
  const int NE = N - 1;

  /* Allocate */
  Node* nodes = (Node*)xcalloc(N, sizeof(Node));
  Elem* elems = (Elem*)xcalloc(NE, sizeof(Elem));
  double* A = (double*)xcalloc(N * N, sizeof(double));
  double* Vb = (double*)xcalloc(N, sizeof(double));
  double* lambda = (double*)xcalloc(N, sizeof(double));

  /* Build geometry */
  build_sphere(a, N, nodes, elems);

  /* Right-hand side: Dirichlet V=1 V on conductor surface at all collocation nodes */
  for (int i = 0; i < N; ++i) Vb[i] = 1.0;

  /* Assemble system */
  fprintf(stderr, "Assembling matrix A (%d x %d)...\n", N, N);
  assemble_A(A, nodes, elems, N, NE);

  /* Solve */
  fprintf(stderr, "Solving A * lambda = V ...\n");
  int rc = solve_linear(N, A, Vb, lambda);
  if (rc != 0) {
    fprintf(stderr, "Linear solve failed (singular matrix)\n");
    return 1;
  }

  /* Report sample of lambda at selected angles */
  printf("# i  theta[deg]  R  Z  lambda[C/m]\n");
  for (int i = 0; i < N; i += N/16) {
    double t = (double)i / (double)(N - 1);
    double th = M_PI * t;
    printf("%4d  %9.4f  % .6e  % .6e  % .6e\n",
           i, th * 180.0 / M_PI, nodes[i].R, nodes[i].Z, lambda[i]);
  }

  /* Total charge and capacitance check for a sphere: Q_exact = 4*pi*eps0*a (for V=1) */
  double Q = total_charge(nodes, elems, lambda, NE);
  double Q_exact = 4.0 * M_PI * eps0 * a;
  printf("\nComputed total charge Q = %.9e C\n", Q);
  printf("Exact sphere charge   Q = %.9e C\n", Q_exact);
  printf("Relative error = %.3e\n", fabs(Q - Q_exact) / (fabs(Q_exact) + 1e-300));

  /* Spot-check potential at an external point (r=a, z=2a); should be ~1 V */
  double rtest = a, ztest = 2.0 * a;
  double Vtest = evaluate_potential(rtest, ztest, nodes, elems, N, NE, lambda);
  printf("\nSpot-check potential at (r=%.3f, z=%.3f): V ~= %.6f V\n", rtest, ztest, Vtest);

  /* after solving A lambda = Vb */
double Er_bem, Ez_bem;
double r_test = 1.5 * a;  /* 外部のテスト点 */
double z_test = 0.0;

/* If you want surface normal just outside, offset slightly: r_test = Rnode + 1e-8*a, z_test likewise */
evaluate_electric_field(r_test, z_test, nodes, elems, N, NE, lambda, &Er_bem, &Ez_bem);

/* analytical for sphere */
double rho = sqrt(r_test*r_test + z_test*z_test);
double E_mag_anal = (a * V0) / (rho * rho); /* V0 is potential (1.0 in your code) */
double Er_anal = E_mag_anal * (r_test / rho);
double Ez_anal = E_mag_anal * (z_test / rho);

/* print compare */
printf("E_r analytical=%.6e bem=%.6e  err=%.6e\n", Er_anal, Er_bem, fabs(Er_anal - Er_bem));
printf("E_z analytical=%.6e bem=%.6e  err=%.6e\n", Ez_anal, Ez_bem, fabs(Ez_anal - Ez_bem));


  /* Clean up */
  free(nodes); free(elems); free(A); free(Vb); free(lambda);
  return 0;
}
