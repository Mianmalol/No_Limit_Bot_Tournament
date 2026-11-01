// Game 2 simulator. Tables are 3 opponents drawn from a field; a strategy's value is
// its mean payout (9/4/2/0, ties split) against them. Rank payoffs don't separate by
// castle, so best responses come from hill climbing, selected on one set of tables
// and reported on a fresh set.
//
//   game2 fp   <iters> <tables> <restarts> <out> [seed]    fictitious play archive
//   game2 br   <field> <tables> <restarts> <top> [seed]    best responses to a field
//   game2 eval <field> <candidates> <tables> [seed]        score candidates on a field
//   game2 places <field> <candidates> <tables> [seed]      score plus odds of each place
//   game2 screen <tables> <prefix> <field>...              every allocation on each field
//
// Field files have one "c1 c2 c3 c4 c5" per line.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define C 5
#define N 100
#define THREADS 8
#define MAX_FIELD 2000000

typedef struct { int8_t c[C]; } Alloc;
typedef struct { Alloc o[3]; } Table;

static const int VALUE[C] = {1, 2, 3, 4, 5};
static const double PAYOUT[4] = {9, 4, 2, 0};
static double PAY[4][5];  // PAY[above][tied incl. self]
static int ZERO_UNAWARDED = 0;  // set by env ZERO_UNAWARDED: nobody gets an all-zero castle

static void init_pay(void) {
    for (int a = 0; a < 4; a++)
        for (int t = 1; a + t <= 4; t++) {
            double s = 0;
            for (int i = a; i < a + t; i++) s += PAYOUT[i];
            PAY[a][t] = s / t;
        }
}

typedef struct { uint64_t s; } Rng;
static inline uint64_t rnd(Rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline int rint_(Rng *r, int n) { return (int)(rnd(r) % (uint64_t)n); }

// Player 0's rank at a table: how many beat them and how many tie (incl. self).
// Points are in 12ths so ties compare exactly.
static inline void rank_at(const Alloc *x, const Table *t, int *above_out, int *tied_out) {
    int pts[4] = {0, 0, 0, 0};
    for (int c = 0; c < C; c++) {
        int s[4] = {x->c[c], t->o[0].c[c], t->o[1].c[c], t->o[2].c[c]};
        int m = s[0];
        for (int i = 1; i < 4; i++) if (s[i] > m) m = s[i];
        if (m == 0 && ZERO_UNAWARDED) continue;
        int k = 0;
        for (int i = 0; i < 4; i++) k += s[i] == m;
        int share = VALUE[c] * 12 / k;
        for (int i = 0; i < 4; i++) if (s[i] == m) pts[i] += share;
    }
    int above = 0, tied = 1;
    for (int i = 1; i < 4; i++) {
        above += pts[i] > pts[0];
        tied += pts[i] == pts[0];
    }
    *above_out = above;
    *tied_out = tied;
}

static inline double score(const Alloc *x, const Table *t) {
    int above, tied;
    rank_at(x, t, &above, &tied);
    return PAY[above][tied];
}

static double mean_score(const Alloc *x, const Table *t, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += score(x, t + i);
    return s / n;
}

static Alloc random_alloc(Rng *r) {
    int cut[4];
    for (int i = 0; i < 4; i++) cut[i] = rint_(r, N + 1);
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            if (cut[j] < cut[i]) { int t = cut[i]; cut[i] = cut[j]; cut[j] = t; }
    Alloc a;
    a.c[0] = cut[0];
    for (int i = 1; i < 4; i++) a.c[i] = cut[i] - cut[i - 1];
    a.c[4] = N - cut[3];
    return a;
}

static void make_tables(Table *t, int n, const Alloc *field, int m, Rng *r) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 3; j++) t[i].o[j] = field[rint_(r, m)];
}

static double hill_climb(Alloc *x, const Table *t, int n) {
    static const int STEPS[6] = {1, 2, 3, 5, 8, 13};
    double best = mean_score(x, t, n);
    for (;;) {
        Alloc top = *x;
        double top_v = best;
        for (int i = 0; i < C; i++)
            for (int j = 0; j < C; j++) {
                if (i == j) continue;
                for (int k = 0; k < 6; k++) {
                    int d = STEPS[k];
                    if (x->c[i] < d) break;
                    Alloc y = *x;
                    y.c[i] -= d;
                    y.c[j] += d;
                    double v = mean_score(&y, t, n);
                    if (v > top_v + 1e-12) { top_v = v; top = y; }
                }
            }
        if (top_v <= best + 1e-12) return best;
        best = top_v;
        *x = top;
    }
}

typedef struct {
    const Table *tables;
    int n;
    const Alloc *field;
    int field_n;
    int restarts;
    uint64_t seed;
    Alloc *out;
    double *val;
} Job;

static void *climb_worker(void *arg) {
    Job *j = arg;
    Rng r = {j->seed};
    for (int i = 0; i < j->restarts; i++) {
        Alloc x = (i % 2 && j->field_n) ? j->field[rint_(&r, j->field_n)] : random_alloc(&r);
        j->val[i] = hill_climb(&x, j->tables, j->n);
        j->out[i] = x;
    }
    return NULL;
}

// Runs restarts across threads; returns how many results were written to out.
static int climb_all(const Table *t, int n, const Alloc *field, int m, int restarts,
                     uint64_t seed, Alloc *out, double *val) {
    pthread_t th[THREADS];
    Job job[THREADS];
    int per = (restarts + THREADS - 1) / THREADS;
    for (int i = 0; i < THREADS; i++) {
        job[i] = (Job){t, n, field, m, per, seed + 1000003ULL * i, out + i * per, val + i * per};
        pthread_create(&th[i], NULL, climb_worker, &job[i]);
    }
    for (int i = 0; i < THREADS; i++) pthread_join(th[i], NULL);
    return per * THREADS;
}

typedef struct { const Alloc *x; int n; const Table *t; int tn; double *out; } EvalJob;

static void *eval_worker(void *arg) {
    EvalJob *e = arg;
    for (int i = 0; i < e->n; i++) e->out[i] = mean_score(e->x + i, e->t, e->tn);
    return NULL;
}

static void eval_all(const Alloc *x, int n, const Table *t, int tn, double *out) {
    pthread_t th[THREADS];
    EvalJob job[THREADS];
    int per = (n + THREADS - 1) / THREADS;
    for (int i = 0; i < THREADS; i++) {
        int start = i * per < n ? i * per : n;
        int count = start + per <= n ? per : n - start;
        job[i] = (EvalJob){x + start, count, t, tn, out + start};
        pthread_create(&th[i], NULL, eval_worker, &job[i]);
    }
    for (int i = 0; i < THREADS; i++) pthread_join(th[i], NULL);
}

static int read_field(const char *path, Alloc *out, int max) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    int a[C], n = 0;
    while (n < max && fscanf(f, "%d %d %d %d %d", a, a + 1, a + 2, a + 3, a + 4) == 5) {
        for (int c = 0; c < C; c++) out[n].c[c] = a[c];
        n++;
    }
    fclose(f);
    return n;
}

static int dedupe(Alloc *x, int n) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int dup = 0;
        for (int j = 0; j < m && !dup; j++) dup = !memcmp(&x[i], &x[j], sizeof(Alloc));
        if (!dup) x[m++] = x[i];
    }
    return m;
}

// Returns the strategy that does best on selection tables; *reported is its value on
// tables that played no part in choosing it.
static Alloc best_of(Alloc *cand, int n, const Alloc *field, int m, int tables, Rng *r,
                     double *reported) {
    n = dedupe(cand, n);
    Table *a = malloc(sizeof(Table) * tables), *b = malloc(sizeof(Table) * tables);
    double *v = malloc(sizeof(double) * n);
    make_tables(a, tables, field, m, r);
    make_tables(b, tables, field, m, r);
    eval_all(cand, n, a, tables, v);
    int top = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[top]) top = i;
    Alloc out = cand[top];
    *reported = mean_score(&out, b, tables);
    free(a); free(b); free(v);
    return out;
}

static int cmd_fp(int iters, int tables, int restarts, const char *out, uint64_t seed) {
    Rng r = {seed};
    Alloc *arch = malloc(sizeof(Alloc) * (iters + 64));
    int m = 0;
    for (; m < 32; m++) arch[m] = random_alloc(&r);
    Table *t = malloc(sizeof(Table) * tables);
    Alloc *cand = malloc(sizeof(Alloc) * (restarts + THREADS));
    double *val = malloc(sizeof(double) * (restarts + THREADS));
    for (int it = 1; it <= iters; it++) {
        make_tables(t, tables, arch, m, &r);
        int n = climb_all(t, tables, arch, m, restarts, rnd(&r), cand, val);
        double fresh;
        Alloc best = best_of(cand, n, arch, m, tables * 8, &r, &fresh);
        arch[m++] = best;
        if (it % 10 == 0 || it == 1) {
            printf("iter %d br %d %d %d %d %d fresh value %.3f (fair 3.75)\n", it, best.c[0],
                   best.c[1], best.c[2], best.c[3], best.c[4], fresh);
            fflush(stdout);
            FILE *f = fopen(out, "w");
            for (int i = 32; i < m; i++)
                fprintf(f, "%d %d %d %d %d\n", arch[i].c[0], arch[i].c[1], arch[i].c[2],
                        arch[i].c[3], arch[i].c[4]);
            fclose(f);
        }
    }
    return 0;
}

static int cmd_br(const char *field_path, int tables, int restarts, int top, uint64_t seed) {
    Alloc *field = malloc(sizeof(Alloc) * MAX_FIELD);
    int m = read_field(field_path, field, MAX_FIELD);
    Rng r = {seed};
    Table *t = malloc(sizeof(Table) * tables);
    make_tables(t, tables, field, m, &r);
    Alloc *cand = malloc(sizeof(Alloc) * (restarts + THREADS));
    double *val = malloc(sizeof(double) * (restarts + THREADS));
    int n = dedupe(cand, climb_all(t, tables, field, m, restarts, rnd(&r), cand, val));
    int fresh_n = tables * 8;
    Table *fresh = malloc(sizeof(Table) * fresh_n);
    make_tables(fresh, fresh_n, field, m, &r);
    double *v = malloc(sizeof(double) * n);
    eval_all(cand, n, fresh, fresh_n, v);
    for (int k = 0; k < top && k < n; k++) {
        int b = -1;
        for (int i = 0; i < n; i++) if (v[i] > -1 && (b < 0 || v[i] > v[b])) b = i;
        printf("%.3f  %d %d %d %d %d\n", v[b], cand[b].c[0], cand[b].c[1], cand[b].c[2],
               cand[b].c[3], cand[b].c[4]);
        v[b] = -2;
    }
    return 0;
}

static int cmd_eval(const char *field_path, const char *cand_path, int tables, uint64_t seed) {
    Alloc *field = malloc(sizeof(Alloc) * MAX_FIELD);
    Alloc *cand = malloc(sizeof(Alloc) * MAX_FIELD);
    int m = read_field(field_path, field, MAX_FIELD);
    int n = read_field(cand_path, cand, MAX_FIELD);
    Rng r = {seed};
    Table *t = malloc(sizeof(Table) * tables);
    make_tables(t, tables, field, m, &r);
    double *v = malloc(sizeof(double) * n);
    eval_all(cand, n, t, tables, v);
    for (int i = 0; i < n; i++)
        printf("%.4f  %d %d %d %d %d\n", v[i], cand[i].c[0], cand[i].c[1], cand[i].c[2],
               cand[i].c[3], cand[i].c[4]);
    return 0;
}

static int cmd_places(const char *field_path, const char *cand_path, int tables, uint64_t seed) {
    Alloc *field = malloc(sizeof(Alloc) * MAX_FIELD);
    Alloc *cand = malloc(sizeof(Alloc) * MAX_FIELD);
    int m = read_field(field_path, field, MAX_FIELD);
    int n = read_field(cand_path, cand, MAX_FIELD);
    Rng r = {seed};
    Table *t = malloc(sizeof(Table) * tables);
    make_tables(t, tables, field, m, &r);
    for (int i = 0; i < n; i++) {
        double p[4] = {0, 0, 0, 0};
        for (int k = 0; k < tables; k++) {
            int above, tied;
            rank_at(&cand[i], &t[k], &above, &tied);
            for (int q = above; q < above + tied; q++) p[q] += 1.0 / tied;
        }
        double s = 9 * p[0] + 4 * p[1] + 2 * p[2];
        printf("%.4f %.4f %.4f %.4f %.4f  %d %d %d %d %d\n", s / tables, p[0] / tables,
               p[1] / tables, p[2] / tables, p[3] / tables, cand[i].c[0], cand[i].c[1],
               cand[i].c[2], cand[i].c[3], cand[i].c[4]);
    }
    return 0;
}

static int all_allocs(Alloc *out) {
    int n = 0;
    for (int a = 0; a <= N; a++)
        for (int b = 0; a + b <= N; b++)
            for (int c = 0; a + b + c <= N; c++)
                for (int d = 0; a + b + c + d <= N; d++)
                    out[n++] = (Alloc){{a, b, c, d, N - a - b - c - d}};
    return n;
}

// Writes <prefix>.alloc (int8 n x 5) and <prefix>.scores (float n x fields).
static int cmd_screen(int tables, const char *prefix, char **fields, int nf) {
    int total = 4598126;
    Alloc *x = malloc(sizeof(Alloc) * total);
    int n = all_allocs(x);
    char path[1024];
    snprintf(path, sizeof path, "%s.alloc", prefix);
    FILE *f = fopen(path, "wb");
    fwrite(x, sizeof(Alloc), n, f);
    fclose(f);
    snprintf(path, sizeof path, "%s.scores", prefix);
    f = fopen(path, "wb");
    Alloc *field = malloc(sizeof(Alloc) * MAX_FIELD);
    Table *t = malloc(sizeof(Table) * tables);
    double *v = malloc(sizeof(double) * n);
    float *row = malloc(sizeof(float) * n);
    for (int k = 0; k < nf; k++) {
        int m = read_field(fields[k], field, MAX_FIELD);
        Rng r = {1000 + (uint64_t)k};
        make_tables(t, tables, field, m, &r);
        eval_all(x, n, t, tables, v);
        for (int i = 0; i < n; i++) row[i] = (float)v[i];
        fwrite(row, sizeof(float), n, f);
        fprintf(stderr, "field %d/%d done\n", k + 1, nf);
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    init_pay();
    ZERO_UNAWARDED = getenv("ZERO_UNAWARDED") != NULL;
    if (argc >= 6 && !strcmp(argv[1], "fp"))
        return cmd_fp(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argv[5],
                      argc > 6 ? strtoull(argv[6], 0, 10) : 1);
    if (argc >= 6 && !strcmp(argv[1], "br"))
        return cmd_br(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                      argc > 6 ? strtoull(argv[6], 0, 10) : 1);
    if (argc >= 5 && !strcmp(argv[1], "eval"))
        return cmd_eval(argv[2], argv[3], atoi(argv[4]), argc > 5 ? strtoull(argv[5], 0, 10) : 1);
    if (argc >= 5 && !strcmp(argv[1], "places"))
        return cmd_places(argv[2], argv[3], atoi(argv[4]), argc > 5 ? strtoull(argv[5], 0, 10) : 1);
    if (argc >= 5 && !strcmp(argv[1], "screen"))
        return cmd_screen(atoi(argv[2]), argv[3], argv + 4, argc - 4);
    fprintf(stderr, "usage: see header of game2.c\n");
    return 1;
}
