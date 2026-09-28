__mlu_func__ void split1dAlign(const size_t count, const uint32_t align_size,
                  size_t &split_begin, size_t &split_num) {
  size_t repeat_align = CEIL_ALIGN(CEIL_DIV(count, taskDim), align_size);
  split_begin = taskId * repeat_align;
  split_num = repeat_align;
  if (split_begin >= count) {
    split_num = 0;
  } else if (split_begin + split_num > count) {
    split_num = count - split_begin;
  }
}