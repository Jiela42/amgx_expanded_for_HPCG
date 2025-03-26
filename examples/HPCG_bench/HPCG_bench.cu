
// this imports are from AMGX (DO NOT TOUCH)
#include "matrix.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "multiply.h"
#include "thrust/fill.h"
#include "amgx_c.h"

// I just copy stuff from amgx_capi.c to be able to run the solver code with my own configs ;)
/* CUDA error macro */
#define CUDA_SAFE_CALL(call) do {                                 \
    cudaError_t err = call;                                         \
    if(cudaSuccess != err) {                                        \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
              __FILE__, __LINE__, cudaGetErrorString( err) );       \
      exit(EXIT_FAILURE);                                           \
    } } while (0)

/* print error message and exit */
void errAndExit(const char *err)
{
    printf("%s\n", err);
    fflush(stdout);
    exit(1);
}

/* print callback (could be customized) */
void print_callback(const char *msg, int length)
{
    printf("%s", msg);
}


// here are my imports
#include <vector>
#include <tuple>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <sys/types.h>


// Here are some MACROS I defined
#define NUM_ITERATIONS 10
#define DO_TESTS 1
#define AULT_NODE "GH200"
#define MATRIX_TYPE "3d_27pt"
#define VERSION_NAME "AMGX"
#define ADDITIONAL_PARAMETERS ""
#define BENCH_SPMV 1
#define BENCH_SYMGS 1
#define BENCH_CG 1
#define RANDOM_SEED 42

// First we have all the generations we might need

std::vector<double> random_vector(int seed, int size)
{
    std::vector<double> vec(size, 0.0);
    srand(seed);
    for (int i = 0; i < size; i++)
    {
        vec[i] = (double)rand() / RAND_MAX;
    }
    return vec;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, std::vector<double>> generate_HPCG_problem(int nx, int ny, int nz)
{
    
    // this is a copy of a generation method in HighPerformanceHPCG_thesis

    int num_rows = nx * ny * nz;
    // int num_cols = nx * ny * nz;

    int nnz = 0;
    // printf("Generating problem with %d rows, %d cols, %d nnz\n", num_rows, num_cols, nnz);

    // return std::make_tuple(std::vector<int>(), std::vector<int>(), std::vector<double>(), std::vector<double>());

    std::vector<int> row_ptr(num_rows + 1, 0);
    std::vector<int> nnz_per_row(num_rows);
    std::vector<int> col_idx;
    std::vector<double> values;
    std::vector<double> y(num_rows, 0.0);

    std::vector<std::vector<int>> col_idx_per_row(num_rows);
    std::vector<std::vector<double>> values_per_row(num_rows);

    for(int ix = 0; ix < nx; ix++){
        for(int iy = 0; iy < ny; iy++){
            for(int iz = 0; iz < nz; iz++){

                int i = ix + nx * iy + nx * ny * iz;
                int nnz_i = 0;

                for (int sz = -1; sz < 2; sz++){
                    if(iz + sz > -1 && iz + sz < nz){
                        for(int sy = -1; sy < 2; sy++){
                            if(iy + sy > -1 && iy + sy < ny){
                                for(int sx = -1; sx < 2; sx++){
                                    if(ix + sx > -1 && ix + sx < nx){
                                        int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                        if(i == j){
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(26.0);
                                        } else {
                                            col_idx_per_row[i].push_back(j);
                                            values_per_row[i].push_back(-1.0);
                                        }
                                            nnz_i++;
                                            nnz++;
                                    }
                                }
                            }
                        }
                    }
                }
                nnz_per_row[i] = nnz_i;
                y[i] = 26.0 - nnz_i;
            }
        }
    }

    for (int i = 0; i < num_rows; i++){
        row_ptr[i + 1] = row_ptr[i] + nnz_per_row[i];

        for (int j = 0; j < nnz_per_row[i]; j++){
            col_idx.push_back(col_idx_per_row[i][j]);
            values.push_back(values_per_row[i][j]);
        }
    }
    // printf("Generated problem with %d rows, %d cols, %d nnz\n", num_rows, num_cols, nnz);
    return std::make_tuple(row_ptr, col_idx, values, y);
}



// copy the timer.cpp from the HighPerformanceHPCG_Thesis repo
class CudaTimer {
public:
    CudaTimer(int nx, int ny, int nz, int nnz, std::string ault_node, std::string matrix_type, std::string version_name, std::string additional_parameters, std::string folder_path);
    ~CudaTimer();
    void startTimer();
    void stopTimer(std::string method_name);
    float getElapsedTime() const;
    void writeCSV(std::string filepath, std::string file_header, std::vector<float> times);
    void writeResultsToCsv();

private:
    cudaEvent_t start, stop;
    float milliseconds;
    int nx, ny, nz, nnz;
    std::string ault_node, matrix_type, version_name, additional_parameters, folder_path;
    std::string base_filename, base_fileheader;
    std::vector<float> CG_times, MG_times, SymGS_times, SPMV_times, Dot_times;
};

CudaTimer::CudaTimer(
        int nx,
        int ny,
        int nz,
        int nnz,
        std::string ault_node,
        std::string matrix_type,
        std::string version_name,
        std::string additional_parameters,
        std::string folder_path
    ) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        this->nx = nx;
        this->ny = ny;
        this->nz = nz;
        this->nnz = nnz;
        this->ault_node = ault_node;
        this->matrix_type = matrix_type;
        this->version_name = version_name;
        this->additional_parameters = additional_parameters;
        this->folder_path = folder_path;
 }

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    writeResultsToCsv();
}

void CudaTimer::startTimer() {
    cudaEventRecord(start, 0);
}

void CudaTimer::stopTimer(std::string method_name) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (method_name == "compute_CG") CG_times.push_back(milliseconds);
    else if (method_name == "compute_MG") MG_times.push_back(milliseconds);
    else if (method_name == "compute_SymGS") SymGS_times.push_back(milliseconds);
    else if (method_name == "compute_SPMV") SPMV_times.push_back(milliseconds);
    else if (method_name == "compute_Dot") Dot_times.push_back(milliseconds);
    else{
        std::cerr << "Invalid method name: " << method_name << std::endl;
        return;
    }
}

void CudaTimer::writeCSV(std::string filepath, std::string file_header, std::vector<float> times){
    // if the vector is empty we don't need to write anything
    if (times.empty()) return;

    // Open the CSV file in append mode
    std::ofstream csv_file(filepath, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filepath << std::endl;
        return;
    }

    // Check if the file is empty and write the header if it is
    std::ifstream infile(filepath);
    infile.seekg(0, std::ios::end);
    if (infile.tellg() == 0) {
        csv_file << file_header << std::endl;
    }
    infile.close();

    // Write the timing results to the CSV file
    for (const auto& time : times) {
        csv_file << time << std::endl;
    }

    // Close the CSV file
    csv_file.close();
}

void CudaTimer::writeResultsToCsv() {
    std::string dim_sting = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);

    base_filename = folder_path
                + version_name + "_"
                + ault_node + "_"
                + matrix_type + "_"
                + ault_node + "_"
                + dim_sting + "_";

    base_fileheader = version_name + ","
                + ault_node + ","
                + matrix_type + ","
                + std::to_string(nx) + "," + std::to_string(ny) + "," + std::to_string(nz) + "," + std::to_string(nnz) + ",";

    writeCSV(base_filename + "CG.csv", base_fileheader + "CG," + additional_parameters, CG_times);
    writeCSV(base_filename + "MG.csv", base_fileheader + "MG," + additional_parameters, MG_times);
    writeCSV(base_filename + "SymGS.csv", base_fileheader + "SymGS," + additional_parameters, SymGS_times);
    writeCSV(base_filename + "SPMV.csv", base_fileheader + "SPMV," + additional_parameters, SPMV_times);
    writeCSV(base_filename + "Dot.csv", base_fileheader + "Dot," + additional_parameters, Dot_times);
}


std::string write_Problem_to_file(
    int nx,
    int ny,
    int nz
){

    // get meta data
    int num_rows = nx * ny * nz;
    int num_cols = nx * ny * nz;

    int num_interior_points = (nx - 2) * (ny - 2) * (nz - 2);
    int num_face_points = 2 * ((nx - 2) * (ny - 2) + (nx - 2) * (nz - 2) + (ny - 2) * (nz - 2));
    int num_edge_points = 4 * ((nx - 2) + (ny - 2) + (nz - 2));
    int num_corner_points = 8;

    int nnz_interior = 27 * num_interior_points;
    int nnz_face = 18 * num_face_points;
    int nnz_edge = 12 * num_edge_points;
    int nnz_corner = 8 * num_corner_points;

    int nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner;

    // create an mtx file
    std::string dims = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);
    std::string filename = "../../examples/HPCG_bench/problem_" + dims + ".mtx";

    // check if the file already exists
    std::ifstream infile(filename);
    if (infile.good()){
        std::cerr << "File already exists: " << filename << std::endl;
        return filename;
    }

    std::ofstream file(filename);
    file << "%%MatrixMarket matrix coordinate real general" << std::endl;
    file << "%%AMGX rhs" << std::endl;
    file << num_rows << " " << num_cols << " " << nnz << std::endl;

    // write the matix using one based indexing (as required by the mtx format)
    for(int i = 0; i < num_rows; i++){
        int row = i + 1;

        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);

        // iterate over the columns
        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < nz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < ny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < nx){
                                int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                int col = j + 1;
                                if(i == j){
                                    file << row << " " << col << " " << 26.0 << std::endl;
                                } else {
                                    file << row << " " << col << " " << -1.0 << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // write the rhs (also with one based indexing)
    for(int i = 0 ; i < num_rows; i++){
        int nnz_i = 0;
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);
        // iterate over the columns to grab the nnz_i
        for (int sz = -1; sz < 2; sz++){
            if(iz + sz > -1 && iz + sz < nz){
                for(int sy = -1; sy < 2; sy++){
                    if(iy + sy > -1 && iy + sy < ny){
                        for(int sx = -1; sx < 2; sx++){
                            if(ix + sx > -1 && ix + sx < nx){
                                nnz_i++;
                            }
                        }
                    }
                }
            }
        }
        double y_value = 26.0 - nnz_i;
        file << y_value << std::endl;
    }
    file.close();

    std::cout << "Problem written to file: " << filename << std::endl;

    return filename;
}

std::string write_spmv_problem_to_file(int nx, int ny, int nz){
      // get meta data
      int num_rows = nx * ny * nz;
      int num_cols = nx * ny * nz;
  
      int num_interior_points = (nx - 2) * (ny - 2) * (nz - 2);
      int num_face_points = 2 * ((nx - 2) * (ny - 2) + (nx - 2) * (nz - 2) + (ny - 2) * (nz - 2));
      int num_edge_points = 4 * ((nx - 2) + (ny - 2) + (nz - 2));
      int num_corner_points = 8;
  
      int nnz_interior = 27 * num_interior_points;
      int nnz_face = 18 * num_face_points;
      int nnz_edge = 12 * num_edge_points;
      int nnz_corner = 8 * num_corner_points;
  
      int nnz = nnz_interior + nnz_face + nnz_edge + nnz_corner;
  
      // create an mtx file
      std::string dims = std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);
      std::string filename = "../../examples/HPCG_bench/spmv_problem_" + dims + ".mtx";
  
      // check if the file already exists
      std::ifstream infile(filename);
    //   if (infile.good()){
    //       std::cerr << "File already exists: " << filename << std::endl;
    //       return filename;
    //   }
  
      std::ofstream file(filename);
      file << "%%MatrixMarket matrix coordinate real general" << std::endl;
      file << "%%AMGX rhs" << std::endl;
      file << num_rows << " " << num_cols << " " << nnz << std::endl;
  
      // write the matix using one based indexing (as required by the mtx format)
      for(int i = 0; i < num_rows; i++){
          int row = i + 1;
  
          int ix = i % nx;
          int iy = (i / nx) % ny;
          int iz = i / (nx * ny);
        
          double row_sum = 0.0;
          // iterate over the columns
          for (int sz = -1; sz < 2; sz++){
              if(iz + sz > -1 && iz + sz < nz){
                  for(int sy = -1; sy < 2; sy++){
                      if(iy + sy > -1 && iy + sy < ny){
                          for(int sx = -1; sx < 2; sx++){
                              if(ix + sx > -1 && ix + sx < nx){
                                  int j = ix + sx + nx * (iy + sy) + nx * ny * (iz + sz);
                                  int col = j + 1;
                                  if(i == j){
                                      file << row << " " << col << " " << 26.0 << std::endl;
                                      row_sum += 26.0;
                                  } else {
                                      file << row << " " << col << " " << -1.0 << std::endl;
                                      row_sum += -1.0;
                                  }
                              }
                          }
                      }
                  }
              }
          }

          if(i < 10){
              printf("Row %d sum: %f\n", i, row_sum);
          }
      }
      
      srand(RANDOM_SEED);
      // write a random vector (this does not need indexing)
      for(int i = 0 ; i < num_rows; i++){
            double v_value = (double)rand() / RAND_MAX;
            file << v_value << std::endl;
      }
      file.close();
  
      std::cout << "Problem written to file: " << filename << std::endl;
  
      return filename;
}

// directory stuff for writing results
bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0777) == 0 || errno == EEXIST;
}

std::string createTimestampedFolder(const std::string base_folder){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", std::localtime(&in_time_t));
    ss << buffer;

    std::string folder_path = base_folder + ss.str();
    create_directory(folder_path);

    return folder_path;
}

void registerParametersforSpMV();

typedef amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> TConfig; // Type for spmv calculation
typedef amgx::Vector<amgx::TemplateConfig<AMGX_host, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt>> VVector_h; // vector type to retrieve result


void bench_spmv1(
    CudaTimer& timer,
    std::vector<double>& a,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    std::vector<double>& values
    ){

    // we don't need to get the original data, because where we store the result needs to start out at zero

    int num_iterations = NUM_ITERATIONS;

    amgx::Resources res;

    amgx::Matrix<TConfig> A_amgx;
    amgx::Vector<TConfig> a_amgx;
    amgx::Vector<TConfig> zeros_amgx;
    A_amgx.setResources(&res);
    a_amgx.setResources(&res);
    zeros_amgx.setResources(&res);

    int nrows = row_ptr.size() - 1;
    int nnz = values.size();
    A_amgx.resize(nrows, nrows, nnz, 1);
    zeros_amgx.resize(nrows);
    a_amgx.resize(nrows);
    // HPCG SymGS requires x to be zero in the beginning
    amgx::thrust::fill(zeros_amgx.begin(), zeros_amgx.end(), 0.);
    
    // set the matrix
    A_amgx.row_offsets.assign(row_ptr.begin(), row_ptr.end());    
    A_amgx.col_indices.assign(col_idx.begin(), col_idx.end());
    A_amgx.values.assign(values.begin(), values.end());
    //set matrix "completeness" flag
    A_amgx.set_initialized(1);

    // vector values
    a_amgx.assign(a.begin(), a.end());

    // first we check correctness
    if(DO_TESTS){
        // AMGX multiply 
        amgx::multiply(A_amgx, a_amgx, zeros_amgx);

        // get the result to host
        VVector_h y_res_h = zeros_amgx;

        // reference check
        std::vector<double> y_res_ref(zeros_amgx.size(), 0.);
        bool err_found = false;
        for (int r = 0; r < zeros_amgx.size(); r++)
        {
            double y_res_ref = 0.;
            for (int c = row_ptr[r]; c < row_ptr[r + 1]; c++)
            {
                y_res_ref += values[c]*a_amgx[col_idx[c]];
            }
            if(r == 0){
                printf("Reference: %f, AMGX: %f\n", y_res_ref, y_res_h[r]);
            }

            if (std::abs(y_res_ref - y_res_h[r]) > 1e-8)
            {
                printf("SPMV Test Failing: Difference in row %d: reference: %f, AMGX: %f\n", r, y_res_ref, y_res_h[r]);
                err_found = true;
            }
        }

        if (err_found)
            {
                printf("SPMV Test Failed\n");
                num_iterations = 0;
            }

        // zero out the result for the next iteration
        amgx::thrust::fill(zeros_amgx.begin(), zeros_amgx.end(), 0.);
    }

    // now we run the benchmark
    for(int i = 0; i < num_iterations; i++){
        timer.startTimer();
        amgx::multiply(A_amgx, a_amgx, zeros_amgx);
        timer.stopTimer("compute_SPMV");

        // zero out the result for the next iteration
        amgx::thrust::fill(zeros_amgx.begin(), zeros_amgx.end(), 0.);
    }
}  

void sequential_symGS(
    std::vector<int>& A_row_ptr,
    std::vector<int>& A_col_idx,
    std::vector<double>& A_values,
    std::vector<double>& x,
    std::vector<double>& y
    )
{

    int num_rows = y.size();
    int diag_value;

    // forward pass
    for (int i = 0; i < num_rows; i++){
        double my_sum = 0.0;
        for (int j = A_row_ptr[i]; j < A_row_ptr[i+1]; j++){
            int col = A_col_idx[j];
            double val = A_values[j];
            my_sum -= val * x[col];
            if(i == col){
                diag_value = val;
            }
        }

        double diag = diag_value;
        double sum = diag * x[i] + y[i] + my_sum;
        x[i] = sum / diag;           

    }

    // backward pass
    for (int i = num_rows-1; i >= 0; i--){
    double my_sum = 0.0;
    for (int j = A_row_ptr[i]; j < A_row_ptr[i+1]; j++){
        int col = A_col_idx[j];
        double val = A_values[j];
        my_sum -= val * x[col];
        if(i == col){
            diag_value = val;
        }
    }

        double diag = diag_value;
        double sum = diag * x[i] + y[i] + my_sum;
        x[i] = sum / diag;

    }
}


void bench_spmv(
    CudaTimer& timer,
    std::string problem_filename
    ){

    int num_iterations = NUM_ITERATIONS;

    for(int i = 0; i < num_iterations; i++){

        // run AMGX symGS

        AMGX_config_handle cfg;
        AMGX_resources_handle rsrc;
        AMGX_matrix_handle A;
        AMGX_vector_handle b, x;

        // AMGX_SOLVE_STATUS status;

        //input matrix and rhs/solution
        int n = 0;
        int bsize_x = 0;
        int bsize_y = 0;
        int sol_size = 0;
        int sol_bsize = 0;

        // use default mode
        AMGX_Mode mode = AMGX_mode_dDDI;

            /* init */
        AMGX_SAFE_CALL(AMGX_initialize());
        /* system */
        AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        

        // create the config from file
        std::string symGS_config = "../../examples/HPCG_bench/empty_config.json";
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, symGS_config.c_str()));
        
        AMGX_resources_create_simple(&rsrc, cfg);
        AMGX_matrix_create(&A, rsrc, mode);
        AMGX_vector_create(&x, rsrc, mode);
        AMGX_vector_create(&b, rsrc, mode);
        // AMGX_solver_create(&solver, rsrc, mode, cfg);

        AMGX_read_system(A, b, x, problem_filename.c_str());
        
        AMGX_matrix_get_size(A, &n, &bsize_x, &bsize_y);
        AMGX_vector_get_size(x, &sol_size, &sol_bsize);

        // this always happens, our problem does not have an initial guess
        if (sol_size == 0 || sol_bsize == 0)
        {
            AMGX_vector_set_zero(x, n, bsize_x);
        }

        timer.startTimer();

        AMGX_matrix_vector_multiply(A, b, x);
        timer.stopTimer("compute_SPMV");

        // download solutiion
        std::vector<double> x_h(n, 0.0);
        AMGX_vector_download(x, x_h.data());

        // print the first 10 values
        for(int i = 0; i < 10; i++){
            printf("from spmv: x[%d]: %f\n", i, x_h[i]);
        }

        // AMGX_solver_destroy(solver);
        AMGX_vector_destroy(x);
        AMGX_vector_destroy(b);
        AMGX_matrix_destroy(A);
        AMGX_resources_destroy(rsrc);
        /* destroy config (need to use AMGX_SAFE_CALL after this point) */
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
        /* shutdown and exit */
        AMGX_SAFE_CALL(AMGX_finalize());
    }

}


void bench_symGS(
    CudaTimer& timer,
    std::string problem_filename
    ){

    int num_iterations = NUM_ITERATIONS;

    for(int i = 0; i < num_iterations; i++){

        // run AMGX symGS

        AMGX_config_handle cfg;
        AMGX_resources_handle rsrc;
        AMGX_matrix_handle A;
        AMGX_vector_handle b, x;
        AMGX_solver_handle solver;

        AMGX_SOLVE_STATUS status;

        //input matrix and rhs/solution
        int n = 0;
        int bsize_x = 0;
        int bsize_y = 0;
        int sol_size = 0;
        int sol_bsize = 0;

        // use default mode
        AMGX_Mode mode = AMGX_mode_dDDI;

            /* init */
        AMGX_SAFE_CALL(AMGX_initialize());
        /* system */
        AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        

        // create the config from file
        std::string symGS_config = "../../examples/HPCG_bench/symGS_config.json";
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, symGS_config.c_str()));
        
        AMGX_resources_create_simple(&rsrc, cfg);
        AMGX_matrix_create(&A, rsrc, mode);
        AMGX_vector_create(&x, rsrc, mode);
        AMGX_vector_create(&b, rsrc, mode);
        AMGX_solver_create(&solver, rsrc, mode, cfg);

        AMGX_read_system(A, b, x, problem_filename.c_str());
        
        AMGX_matrix_get_size(A, &n, &bsize_x, &bsize_y);
        AMGX_vector_get_size(x, &sol_size, &sol_bsize);

        // this always happens, our problem does not have an initial guess
        if (sol_size == 0 || sol_bsize == 0)
        {
            AMGX_vector_set_zero(x, n, bsize_x);
        }

        // add max_iters to the config
        int max_iters = 1;
        AMGX_config_add_parameters(&cfg, ("config_version=2, default:max_iters=" + std::to_string(max_iters)).c_str());

        timer.startTimer();
        /* solver setup */
        AMGX_solver_setup(solver, A);
        /* solver solve */
        AMGX_solver_solve(solver, b, x);
        timer.stopTimer("compute_SymGS");

        AMGX_solver_get_status(solver, &status);

        printf("SymGS Solver status: %d\n", status);

        // download solutiion
        std::vector<double> x_h(n, 0.0);
        AMGX_vector_download(x, x_h.data());

        // print the first 10 values
        for(int i = 0; i < 10; i++){
            printf("x[%d]: %f\n", i, x_h[i]);
        }

        AMGX_solver_destroy(solver);
        AMGX_vector_destroy(x);
        AMGX_vector_destroy(b);
        AMGX_matrix_destroy(A);
        AMGX_resources_destroy(rsrc);
        /* destroy config (need to use AMGX_SAFE_CALL after this point) */
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
        /* shutdown and exit */
        AMGX_SAFE_CALL(AMGX_finalize());
    }

}


void bench_CG(
    CudaTimer& timer,
    std::string problem_filename
    ){

    int num_iterations = NUM_ITERATIONS;

    for(int i = 0; i < num_iterations; i++){

        // run AMGX symGS

        AMGX_config_handle cfg;
        AMGX_resources_handle rsrc;
        AMGX_matrix_handle A;
        AMGX_vector_handle b, x;
        AMGX_solver_handle solver;

        AMGX_SOLVE_STATUS status;

        //input matrix and rhs/solution
        int n = 0;
        int bsize_x = 0;
        int bsize_y = 0;
        int sol_size = 0;
        int sol_bsize = 0;

        // use default mode
        AMGX_Mode mode = AMGX_mode_dDDI;

            /* init */
        AMGX_SAFE_CALL(AMGX_initialize());
        /* system */
        AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        

        // create the config from file
        std::string CG_config = "../../examples/HPCG_bench/CG_config.json";
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, CG_config.c_str()));
        
        AMGX_resources_create_simple(&rsrc, cfg);
        AMGX_matrix_create(&A, rsrc, mode);
        AMGX_vector_create(&x, rsrc, mode);
        AMGX_vector_create(&b, rsrc, mode);
        AMGX_solver_create(&solver, rsrc, mode, cfg);

        AMGX_read_system(A, b, x, problem_filename.c_str());
        
        AMGX_matrix_get_size(A, &n, &bsize_x, &bsize_y);
        AMGX_vector_get_size(x, &sol_size, &sol_bsize);

        // this always happens, our problem does not have an initial guess
        if (sol_size == 0 || sol_bsize == 0)
        {
            AMGX_vector_set_zero(x, n, bsize_x);
        }

        timer.startTimer();
        /* solver setup */
        AMGX_solver_setup(solver, A);
        /* solver solve */
        AMGX_solver_solve(solver, b, x);
        timer.stopTimer("compute_CG");
        /* example of how to change parameters between non-linear iterations */
        //AMGX_config_add_parameters(&cfg, "config_version=2, default:tolerance=1e-12");
        //AMGX_solver_solve(solver, b, x);
        AMGX_solver_get_status(solver, &status);

        printf("CG Solver status: %d\n", status);

        AMGX_solver_destroy(solver);
        AMGX_vector_destroy(x);
        AMGX_vector_destroy(b);
        AMGX_matrix_destroy(A);
        AMGX_resources_destroy(rsrc);
        /* destroy config (need to use AMGX_SAFE_CALL after this point) */
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
        /* shutdown and exit */
        AMGX_SAFE_CALL(AMGX_finalize());
    }

}

void run_amgx_benchmark(int nx, int ny, int nz, std::string folder_path){

    // generate problem
    std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, std::vector<double>> problem = generate_HPCG_problem(nx, ny, nz);

    std::vector<int>& row_ptr = std::get<0>(problem);
    std::vector<int>& col_idx = std::get<1>(problem);
    std::vector<double>& values = std::get<2>(problem);
    std::vector<double>& y = std::get<3>(problem);
    std::vector<double> x (nx * ny * nz, 0.0);
    std::vector<double> a = random_vector(RANDOM_SEED, nx * ny * nz);

    // write the problem to a file
    std::string file_name = write_Problem_to_file(nx, ny, nz);
    std::string spmv_file_name = write_spmv_problem_to_file(nx, ny, nz);

    int nnz = values.size();

    // now we gotta greb the timer
    CudaTimer* timer = new CudaTimer(nx, ny, nz, nnz, AULT_NODE, MATRIX_TYPE, VERSION_NAME, ADDITIONAL_PARAMETERS, folder_path);


    // appearently the spmv requires the results to be zero initialized
    if (BENCH_SPMV){
        bench_spmv(
            *timer,
            spmv_file_name
            // a,
            // row_ptr,
            // col_idx,
            // values
        );
    }

    if (BENCH_SYMGS){
        bench_symGS(
            *timer,
            file_name
        );
    }
    if (BENCH_CG){
        bench_CG(
            *timer,
            file_name
        );
    }


    delete timer;
}

int main(int argc, char* argv[])
{
    // Initialization
    cudaSetDevice(0);
    // register required AMGX parameters 
    registerParametersforSpMV();

    // generate a timestamped folder
    std::string base_path = "../../../HighPerformanceHPCG_Thesis/timing_results/";
    base_path = "../../../HighPerformanceHPCG_Thesis/dummy_timing_results/";

    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    std::cout << "Starting Benchmark" << std::endl;
    run_amgx_benchmark(8, 8, 8, folder_path);
    // run_amgx_benchmark(16, 16, 16, folder_path);
    // run_amgx_benchmark(24, 24, 24, folder_path);
    // run_amgx_benchmark(32, 32, 32, folder_path);
    // run_amgx_benchmark(64, 64, 64, folder_path);
    // run_amgx_benchmark(128, 64, 64, folder_path);
    // run_amgx_benchmark(128, 128, 64, folder_path);
    // run_amgx_benchmark(128, 128, 128, folder_path);
    // run_amgx_benchmark(256, 128, 128, folder_path);

    std::cout << "Benchmark Finished" << std::endl;

       
    return 0;
}


// Routine to register some of the AMGX parameters manually
// Typically if you want to use AMGX solver you should call core::initialize() which will cover this initialization, 
// however for spmv alone there are only few parameters needed and no need to initialize AMGX core solvers.
void registerParametersforSpMV()
{
    using namespace amgx;
    std::vector<int> bool_flag_values;
    bool_flag_values.push_back(0);
    bool_flag_values.push_back(1);
    //Register Exception Handling Parameter
    AMG_Config::registerParameter<int>("exception_handling", "a flag that forces internal exception processing instead of returning error codes(1:internal, 0:external)", 0, bool_flag_values);
    //Register System Parameters (memory pools)
    AMG_Config::registerParameter<size_t>("device_mem_pool_size", "size of the device memory pool in bytes", 256 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_consolidation_pool_size", "size of the device memory pool for root partition in bytes", 256 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_max_alloc_size", "maximum size of a single allocation in the device memory pool in bytes", 20 * 1024 * 1024);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_factor", "over allocation for large buffers (in %% -- a value of X will lead to 100+X%% allocations)", 10);
    AMG_Config::registerParameter<size_t>("device_alloc_scaling_threshold", "buffers smaller than that threshold will NOT be scaled", 16 * 1024);
    AMG_Config::registerParameter<size_t>("device_mem_pool_size_limit", "size of the device memory pool in bytes. 0 - no limit", 0);
    //Register System Parameters (asynchronous framework)
    AMG_Config::registerParameter<int>("num_streams", "number of additional CUDA streams / threads used for async execution", 0);
    AMG_Config::registerParameter<int>("serialize_threads", "flag that enables thread serialization for debugging <0|1>", 0, bool_flag_values);
    AMG_Config::registerParameter<int>("high_priority_stream", "flag that enables high priority CUDA stream <0|1>", 0, bool_flag_values);
    //Register System Parameters (in distributed setting)
    std::vector<std::string> communicator_values;
    communicator_values.push_back("MPI");
    communicator_values.push_back("MPI_DIRECT");
    AMG_Config::registerParameter<std::string>("communicator", "type of communicator <MPI|MPI_DIRECT>", "MPI");
    std::vector<ViewType> viewtype_values;
    viewtype_values.push_back(INTERIOR);
    viewtype_values.push_back(OWNED);
    viewtype_values.push_back(FULL);
    viewtype_values.push_back(ALL);
    AMG_Config::registerParameter<ViewType>("separation_interior", "separation for latency hiding and coloring/smoothing <ViewType>", INTERIOR, viewtype_values);
    AMG_Config::registerParameter<ViewType>("separation_exterior", "limit of calculations for coloring/smoothing <ViewType>", OWNED, viewtype_values);
    AMG_Config::registerParameter<int>("min_rows_latency_hiding", "number of rows at which to disable latency hiding, negative value means latency hiding is completely disabled", -1);
    AMG_Config::registerParameter<int>("matrix_halo_exchange", "0 - No halo exchange on lower levels, 1 - just diagonal values, 2 - full", 0);
    AMG_Config::registerParameter<std::string>("solver", "", "");
    AMG_Config::registerParameter<int>("verbosity_level", "verbosity level for output, 3 - custom print-outs <0|1|2|3>", 3);   
}
