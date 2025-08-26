#pragma once
#include <iostream>
#include <cstdio>
#include <type_traits>
#include <cstdint>

template <typename>
inline constexpr bool always_false = false;

template <typename T>
__mlu_func__ __mlu_host__ void print_arg(const T &arg) {
  if constexpr (std::is_same_v<int, T>) {
    printf("%d", arg);
  } else if constexpr (std::is_same_v<int64_t, T>) {
    printf("%ld", arg);
  } else if constexpr (std::is_same_v<bool, T>) {
    printf("%d", arg);
  } else if constexpr (std::is_same_v<size_t, T>) {
    printf("%zu", arg);
  } else if constexpr (std::is_same_v<uint32_t, T>) {
    printf("%u", arg);
  } else if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
    printf("%f", arg);
  } else if constexpr (std::is_same_v<half, T>) {
#if defined(__BANG__)  // device 编译
    printf("%hf", arg);
#else  // host 编译
    printf("%f", static_cast<float>(arg));
#endif
  } else if constexpr (std::is_same_v<bfloat16_t, T>) {
    printf("%f", static_cast<float>(arg));
  } else if constexpr (std::is_same_v<int8_t, T> ||
                       std::is_same_v<uint8_t, T>) {
    printf("%d", static_cast<int>(arg));
  } else if constexpr (std::is_pointer_v<T>) {
    printf("%p", arg);
  } else if constexpr (std::is_same_v<wchar_t, T>) {
    printf("%lc", arg);
  } else {
    static_assert(always_false<T>, "Unsupported type");
  }
}

template <typename T, typename... Args>
__mlu_func__ __mlu_host__ void print_args(const T &first, const Args &...rest) {
  print_arg(first);
  if constexpr (sizeof...(rest) > 0) {
    printf(",\t");
    print_args(rest...);
  }
}

// 小端序，按位打印单个参数
template <typename T>
__mlu_func__ __mlu_host__ void print_arg_bit(const T &arg) {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&arg);
  for (size_t i = 0; i < sizeof(T); ++i) {
    for (int j = 7; j >= 0; --j) {
      printf("%d", (bytes[i] >> j) & 1);
    }
    if (i < sizeof(T) - 1) printf(" ");
  }
}

// 多参数按位打印
template <typename T, typename... Args>
__mlu_func__ __mlu_host__ void print_args_bit(const T &first,
                                              const Args &...rest) {
  print_arg_bit(first);
  if constexpr (sizeof...(rest) > 0) {
    printf(",\t");
    print_args_bit(rest...);
  }
}

// 按值打印宏
#define TASK_PRINT(id, ...)     \
  if (taskId == id) {           \
    printf(#__VA_ARGS__ ":\n"); \
    print_args(__VA_ARGS__);    \
    printf("\n");               \
  }

#define DEVICE_PRINT(...)                             \
  printf("taskId: %d, line: %d\n", taskId, __LINE__); \
  printf(#__VA_ARGS__ ":\n");                         \
  print_args(__VA_ARGS__);                            \
  printf("\n");

#define HOST_PRINT(...)       \
  printf(#__VA_ARGS__ ":\n"); \
  print_args(__VA_ARGS__);    \
  printf("\n");

// 按位打印宏
#define TASK_PRINT_BIT(id, ...)       \
  if (taskId == id) {                 \
    printf(#__VA_ARGS__ " (bit):\n"); \
    print_args_bit(__VA_ARGS__);      \
    printf("\n");                     \
  }

#define DEVICE_PRINT_BIT(...)                         \
  printf("taskId: %d, line: %d\n", taskId, __LINE__); \
  printf(#__VA_ARGS__ " (bit):\n");                   \
  print_args_bit(__VA_ARGS__);                        \
  printf("\n");

#define HOST_PRINT_BIT(...)         \
  printf(#__VA_ARGS__ " (bit):\n"); \
  print_args_bit(__VA_ARGS__);      \
  printf("\n");
