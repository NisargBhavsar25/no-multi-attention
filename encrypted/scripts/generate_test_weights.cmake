
# This script generates dummy weight files for testing the encrypted transformer
file(MAKE_DIRECTORY /home/user/btp_nisarg_zaid/no-multi-attention/encrypted/build/model)

# Function to create a dummy weight file
function(create_dummy_weight_file filename num_layers element_count)
    # Create binary file with:
    # - 4 bytes: number of layers (uint32_t)
    # - For each layer:
    #   - 4 bytes: number of elements (uint32_t)
    #   - 8 * num_elements bytes: elements (double)
    
    file(WRITE /home/user/btp_nisarg_zaid/no-multi-attention/encrypted/build/model/ "dummy")
    
    # In a real script, this would write actual binary data
    # For now, we just create empty files as placeholders
    message(STATUS "Created dummy weight file: /home/user/btp_nisarg_zaid/no-multi-attention/encrypted/build/model/")
endfunction()

# Create weight files
create_dummy_weight_file(wq.bin 1 16384)
create_dummy_weight_file(wk.bin 1 16384)
create_dummy_weight_file(wv.bin 1 16384)
create_dummy_weight_file(wo.bin 1 16384)
create_dummy_weight_file(ff1.bin 1 16384)
create_dummy_weight_file(ff2.bin 1 16384)
