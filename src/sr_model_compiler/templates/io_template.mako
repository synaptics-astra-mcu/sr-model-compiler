
#include <cstdint>

#include "inference_attributes.hpp"

namespace ${namespace} {

% for i, input_data in enumerate(input_data_list):
static int8_t IFM_BUF_ATTRIBUTE input_data${i}[${input_data_size_list[i]}] = {
${input_data},
};

% endfor
% for i, output_data in enumerate(output_data_list):
static int8_t LABELS_ATTRIBUTE output_data${i}[${output_data_size_list[i]}] = {
${output_data},
};

% endfor

int8_t* get_user_input_buffer(int index) {
    switch (index) {
    % for i in range(len(input_data_list)):
        case ${i}:
            return input_data${i};
    % endfor
        default:
            return nullptr;
    }
}

int8_t* get_expected_output_buffer(int index) {
    switch (index) {
    % for i in range(len(output_data_list)):
        case ${i}:
            return output_data${i};
    % endfor
        default:
            return nullptr;
    }
}

}  /* namespace ${namespace} */
