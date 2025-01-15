
// this imports are from AMGX (DO NOT TOUCH)
#include "matrix.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "multiply.h"
#include "thrust/fill.h"

// here are my imports
#include <vector>
#include <tuple>
#include <fstream>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <sys/types.h>


// Here are some MACROS I defined
#define NUM_ITERATIONS 10
#define DO_TESTS 0
#define AULT_NODE "41-44"
#define MATRIX_TYPE "3d_27pt"
#define VERSION_NAME "AMGX"
#define ADDITIONAL_PARAMETERS ""
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

float CudaTimer::getElapsedTime() const {
    return milliseconds;
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


void spmv_example(amgx::Resources& res)
{
    // TemplateConfig parameters:
    // calculate on device
    // double storage for matrix values
    // double storage for vector values
    // integer for indices values
    //
    // see include/basic_types.h for details

    int nx = 8;
    int ny = 8;
    int nz = 8;

    // generate problem
    std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, std::vector<double>> problem = generate_HPCG_problem(nx, ny, nz);

    std::vector<int>& row_ptr = std::get<0>(problem);
    std::vector<int>& col_idx = std::get<1>(problem);
    std::vector<double>& values = std::get<2>(problem);
    std::vector<double>& y = std::get<3>(problem);

    std::vector<double> x = random_vector(RANDOM_SEED, nx * ny * nz);


    typedef amgx::TemplateConfig<AMGX_device, AMGX_vecDouble, AMGX_matDouble, AMGX_indInt> TConfig; // Type for spmv calculation

    amgx::Matrix<TConfig> A_amgx;
    amgx::Vector<TConfig> a_amgx;
    amgx::Vector<TConfig> x_amgx;
    amgx::Vector<TConfig> y_amgx;
    A_amgx.setResources(&res);
    a_amgx.setResources(&res);
    x_amgx.setResources(&res);
    y_amgx.setResources(&res);

    int nrows = nx * ny * nz;
    int nnz = values.size();
    A_amgx.resize(nrows, nrows, nnz, 1);
    x_amgx.resize(nrows);
    y_amgx.resize(nrows);
    amgx::thrust::fill(y_amgx.begin(), y_amgx.end(), 0.);
    
    // set the matrix
    A_amgx.row_offsets.assign(row_ptr.begin(), row_ptr.end());    
    A_amgx.col_indices.assign(col_idx.begin(), col_idx.end());
    A_amgx.values.assign(values.begin(), values.end());
    //set matrix "completeness" flag
    A_amgx.set_initialized(1);

    // vector values
    x_amgx.assign(x.begin(), x.end());
    

    // AMGX multiply 
    amgx::multiply(A_amgx, x_amgx, y_amgx);

    // get the result to host
    VVector_h y_res_h = y_amgx;

    // reference check
    std::vector<double> y_res_ref(nrows, 0.);
    bool err_found = false;
    for (int r = 0; r < nrows; r++)
    {
        double y_res_ref = 0.;
        for (int c = row_ptr[r]; c < row_ptr[r + 1]; c++)
        {
            y_res_ref += values[c]*x[col_idx[c]];
        }
        if (std::abs(y_res_ref - y_res_h[r]) > 1e-8)
        {
            printf("Difference in row %d: reference: %f, AMGX: %f\n", r, y_res_ref, y_res_h[r]);
            err_found = true;
        }
    }

    if (!err_found)
        printf("Done!\n");
}

void bench_spmv(
    CudaTimer& timer,
    amgx::Matrix<TConfig>& A_amgx,
    amgx::Vector<TConfig>& a_amgx,
    amgx::Vector<TConfig>& zeros_amgx,
    std::vector<double>& a,
    std::vector<int>& row_ptr,
    std::vector<int>& col_idx,
    std::vector<double>& values
    ){

    // we don't need to get the original data, because where we store the result needs to start out at zero

    int num_iterations = NUM_ITERATIONS;

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
            if (std::abs(y_res_ref - y_res_h[r]) > 1e-8)
            {
                printf("Difference in row %d: reference: %f, AMGX: %f\n", r, y_res_ref, y_res_h[r]);
                err_found = true;
            }
        }

        if (!err_found)
            {
                std::cout << "Test Successful" << std::endl;
            }
        else
            {
                printf("Test Failed\n");
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

void run_amgx_benchmark(int nx, int ny, int nz, std::string folder_path){

    // resources object
    amgx::Resources res;

    // make sure we perform any AMGX functionality within Resources lifetime
    // spmv_example(res);


    // generate problem
    std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, std::vector<double>> problem = generate_HPCG_problem(nx, ny, nz);

    std::vector<int>& row_ptr = std::get<0>(problem);
    std::vector<int>& col_idx = std::get<1>(problem);
    std::vector<double>& values = std::get<2>(problem);
    std::vector<double>& y = std::get<3>(problem);
    std::vector<double> a = random_vector(RANDOM_SEED, nx * ny * nz);

    amgx::Matrix<TConfig> A_amgx;
    amgx::Vector<TConfig> a_amgx;
    amgx::Vector<TConfig> x_amgx;
    amgx::Vector<TConfig> y_amgx;
    amgx::Vector<TConfig> zeros_amgx;
    A_amgx.setResources(&res);
    a_amgx.setResources(&res);
    x_amgx.setResources(&res);
    y_amgx.setResources(&res);
    zeros_amgx.setResources(&res);

    int nrows = nx * ny * nz;
    int nnz = values.size();
    A_amgx.resize(nrows, nrows, nnz, 1);
    x_amgx.resize(nrows);
    y_amgx.resize(nrows);
    zeros_amgx.resize(nrows);
    a_amgx.resize(nrows);
    // HPCG SymGS requires x to be zero in the beginning
    amgx::thrust::fill(x_amgx.begin(), x_amgx.end(), 0.);
    amgx::thrust::fill(zeros_amgx.begin(), zeros_amgx.end(), 0.);
    
    // set the matrix
    A_amgx.row_offsets.assign(row_ptr.begin(), row_ptr.end());    
    A_amgx.col_indices.assign(col_idx.begin(), col_idx.end());
    A_amgx.values.assign(values.begin(), values.end());
    //set matrix "completeness" flag
    A_amgx.set_initialized(1);

    // vector values
    y_amgx.assign(y.begin(), y.end());
    a_amgx.assign(a.begin(), a.end());

    // now we gotta greb the timer
    CudaTimer* timer = new CudaTimer(nx, ny, nz, nnz, AULT_NODE, MATRIX_TYPE, VERSION_NAME, ADDITIONAL_PARAMETERS, folder_path);


    // functions to write:
        // appearently the spmv requires the results to be zero initialized
        // bench spmv
        bench_spmv(
            *timer,
            A_amgx,
            a_amgx,
            zeros_amgx,
            a,
            row_ptr,
            col_idx,
            values
        );
        // bench SymGS

    delete timer;
}

int main(int argc, char* argv[])
{
    // Initialization
    cudaSetDevice(0);
    // register required AMGX parameters 
    registerParametersforSpMV();

    // generate a timestamped folder
    std::string base_path = "../../HighPerformanceHPCG_Thesis/timing_results/";
    // base_path = "../../HighPerformanceHPCG_Thesis/dummy_timing_results/";

    std::string folder_path = createTimestampedFolder(base_path);
    folder_path += "/";

    std::cout << "Starting Benchmark" << std::endl;
    run_amgx_benchmark(8, 8, 8, folder_path);
    run_amgx_benchmark(16, 16, 16, folder_path);
    run_amgx_benchmark(24, 24, 24, folder_path);
    run_amgx_benchmark(32, 32, 32, folder_path);
    run_amgx_benchmark(64, 64, 64, folder_path);
    run_amgx_benchmark(128, 64, 64, folder_path);
    run_amgx_benchmark(128, 128, 64, folder_path);
    std::cout << "did 64" << std::endl;
    run_amgx_benchmark(128, 128, 128, folder_path);
    run_amgx_benchmark(256, 128, 128, folder_path);

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
