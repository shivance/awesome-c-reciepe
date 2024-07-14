#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double tanh_activation(double x) {
    return tanh(x);
}

// LSTM Cell structure
typedef struct {
    int input_size;
    int hidden_size;

    double *Wf, *Wi, *Wo, *Wc;
    double *Uf, *Ui, *Uo, *Uc;
    double *bf, *bi, *bo, *bc;

    double *hidden_state;
    double *cell_state;
} LSTMCell;

// Initialize LSTM Cell
void initialize_lstm(LSTMCell *cell, int input_size, int hidden_size) {
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;

    cell->Wf = (double *)malloc(input_size * hidden_size * sizeof(double));
    cell->Wi = (double *)malloc(input_size * hidden_size * sizeof(double));
    cell->Wo = (double *)malloc(input_size * hidden_size * sizeof(double));
    cell->Wc = (double *)malloc(input_size * hidden_size * sizeof(double));

    cell->Uf = (double *)malloc(hidden_size * hidden_size * sizeof(double));
    cell->Ui = (double *)malloc(hidden_size * hidden_size * sizeof(double));
    cell->Uo = (double *)malloc(hidden_size * hidden_size * sizeof(double));
    cell->Uc = (double *)malloc(hidden_size * hidden_size * sizeof(double));

    cell->bf = (double *)malloc(hidden_size * sizeof(double));
    cell->bi = (double *)malloc(hidden_size * sizeof(double));
    cell->bo = (double *)malloc(hidden_size * sizeof(double));
    cell->bc = (double *)malloc(hidden_size * sizeof(double));

    cell->hidden_state = (double *)calloc(hidden_size, sizeof(double));
    cell->cell_state = (double *)calloc(hidden_size, sizeof(double));
}

// Single LSTM forward pass
void lstm_forward(LSTMCell *cell, double *input) {
    double *f = (double *)malloc(cell->hidden_size * sizeof(double));
    double *i = (double *)malloc(cell->hidden_size * sizeof(double));
    double *o = (double *)malloc(cell->hidden_size * sizeof(double));
    double *c_bar = (double *)malloc(cell->hidden_size * sizeof(double));

    // Forget gate
    for (int j = 0; j < cell->hidden_size; j++) {
        f[j] = sigmoid(input[j] * cell->Wf[j] + cell->hidden_state[j] * cell->Uf[j] + cell->bf[j]);
    }

    // Input gate
    for (int j = 0; j < cell->hidden_size; j++) {
        i[j] = sigmoid(input[j] * cell->Wi[j] + cell->hidden_state[j] * cell->Ui[j] + cell->bi[j]);
    }

    // Candidate cell state
    for (int j = 0; j < cell->hidden_size; j++) {
        c_bar[j] = tanh_activation(input[j] * cell->Wc[j] + cell->hidden_state[j] * cell->Uc[j] + cell->bc[j]);
    }

    // Output gate
    for (int j = 0; j < cell->hidden_size; j++) {
        o[j] = sigmoid(input[j] * cell->Wo[j] + cell->hidden_state[j] * cell->Uo[j] + cell->bo[j]);
    }

    // New cell state
    for (int j = 0; j < cell->hidden_size; j++) {
        cell->cell_state[j] = f[j] * cell->cell_state[j] + i[j] * c_bar[j];
    }

    // New hidden state
    for (int j = 0; j < cell->hidden_size; j++) {
        cell->hidden_state[j] = o[j] * tanh_activation(cell->cell_state[j]);
    }

    free(f);
    free(i);
    free(o);
    free(c_bar);
}

// Cleanup
void free_lstm(LSTMCell *cell) {
    free(cell->Wf);
    free(cell->Wi);
    free(cell->Wo);
    free(cell->Wc);

    free(cell->Uf);
    free(cell->Ui);
    free(cell->Uo);
    free(cell->Uc);

    free(cell->bf);
    free(cell->bi);
    free(cell->bo);
    free(cell->bc);

    free(cell->hidden_state);
    free(cell->cell_state);
}

int main() {
    int input_size = 4;
    int hidden_size = 3;

    LSTMCell cell;
    initialize_lstm(&cell, input_size, hidden_size);

    double input[4] = {1.0, 2.0, 3.0, 4.0};

    lstm_forward(&cell, input);

    // Print the hidden state
    printf("Hidden state:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("%f ", cell.hidden_state[i]);
    }
    printf("\n");

    free_lstm(&cell);

    return 0;
}

