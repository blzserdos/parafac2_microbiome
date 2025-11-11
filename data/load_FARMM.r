# --- 1. Load reticulate ---
# Make sure you have the 'reticulate' package installed
# install.packages("reticulate")
library(reticulate)
# --- 2. Load the data from the URLs ---
# load(url(...)) downloads the file and loads its contents into the global environment.
# We expect this to create objects named 'X_array' and 'df_samples'.
cat("Attempting to load X_array.RData from GitHub...\n")
load(url("https://raw.githubusercontent.com/syma-research/microTensor/main/data/FARMM/X_array.RData"))
cat("Loaded X_array successfully.\n")

cat("Attempting to load df_samples.RData from GitHub...\n")
load(url("https://raw.githubusercontent.com/syma-research/microTensor/main/data/FARMM/df_samples.RData"))


# --- 3. Import Python's numpy module ---
# Check if numpy is available in your Python environment
if (!py_module_available("numpy")) {
  cat("Numpy not found. Attempting to install it using reticulate...\n")
  tryCatch({
    py_install("numpy")
  }, error = function(e) {
    stop("Failed to install numpy. Please install it in your Python environment manually (e.g., 'pip install numpy') and try again.\n", e)
  })
}

np <- import("numpy")

# --- 4. Save the R objects to files ---
# Save the count tensor (X_array) to a .npy file
cat("Saving X_array.npy...\n")
np$save("FARMM_data.npy", r_to_py(X_array))

# Save the row names (taxa) from X_array to a .csv file
write.csv(dimnames(X_array)[[1]], "FARMM_data_taxonomy.csv", row.names = FALSE, quote = FALSE)

# Save the sample metadata (df_samples) to a .csv file
write.csv(df_samples, "FARMM_metadata.csv", row.names = FALSE)
