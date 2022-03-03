#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#define SLOW_LIGHT (0)

// Model-specific definitions and globals
#define KRHO 0
#define UU   1
#define U1   2
#define U2   3
#define U3   4
#define B1   5
#define B2   6
#define B3   7
#define KEL  8
#define KTOT 9
// These two will never be used simultaneously,
// and never with KEL.
#define TFLK 8  // temperature of fluid in Kelvin
#define THF  8  // fluid temperature in me c^2
#define THETAE_HOWES 8
#define THETAE_ROWAN 9
#define THETAE_WERNER 10
extern double DTd;
extern double sigma_cut;

#endif // MODEL_PARAMS_H
