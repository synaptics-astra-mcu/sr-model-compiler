

namespace ${namespace} {

constexpr int kNumberOperators = ${number_of_ops};
tflite::MicroMutableOpResolver<kNumberOperators> micro_op_resolver;
static bool initialized = false;

tflite::MicroOpResolver& get_resolver(void)
{
  if (initialized == false)
  { 
    % for operator in operators:
      micro_op_resolver.${operator}();
    % endfor
      initialized = true;
  }

  return micro_op_resolver;
}

}  /* namespace ${namespace} */