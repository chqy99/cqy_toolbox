#pragma once
#include <iostream>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

// 区分 mlu 文件和 cpp 文件
#ifdef __BANG__
#define FUNCTION_ATTRIBUTE __mlu_func__ __mlu_host__
#else
#define FUNCTION_ATTRIBUTE
#endif

#ifdef __BANG_ARCH__
// 区分 device 侧和 host 侧
__mlu_device__ void print_taskId() { printf("taskId: %d, ", taskId); }
__mlu_host__ void print_taskId() {}
#define PRINT_HEADER(tag, ...)            \
  print_taskId();                         \
  printf("line: %d %s\n", __LINE__, tag); \
  printf(#__VA_ARGS__ ":\n");
#else
#define PRINT_HEADER(tag, ...)            \
  printf("line: %d %s\n", __LINE__, tag); \
  printf(#__VA_ARGS__ ":\n");
#endif

template <typename>
inline constexpr bool always_false = false;

template <typename T>
FUNCTION_ATTRIBUTE void print_arg(const T &arg) {
  if constexpr (std::is_same_v<int, T>) {
    printf("%d", arg);
  } else if constexpr (std::is_same_v<int64_t, T>) {
    printf("%ld", arg);
  } else if constexpr (std::is_same_v<bool, T>) {
    printf("%d", arg);
  } else if constexpr (std::is_same_v<size_t, T>) {
    printf("%lu", arg);
  } else if constexpr (std::is_same_v<uint32_t, T>) {
    printf("%u", arg);
  } else if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
    printf("%f", arg);
  }
#ifdef __BANG__
  else if constexpr (std::is_same_v<half, T>) {
    printf("%hf", arg);
  } else if constexpr (std::is_same_v<bfloat16_t, T>) {
    printf("%f", static_cast<float>(arg));
  }
#endif
  else if constexpr (std::is_same_v<int8_t, T> || std::is_same_v<uint8_t, T>) {
    printf("%d", static_cast<int>(arg));
  } else if constexpr (std::is_pointer_v<T>) {
    printf("%p", arg);
  } else if constexpr (std::is_same_v<wchar_t, T>) {
    printf("%lc", arg);
  } else if constexpr (std::is_same_v<const char *, T>) {
    printf("%s", arg);
  } else if constexpr (std::is_array_v<T> &&
                       std::is_same_v<std::remove_extent_t<T>, char>) {
    printf("%s", arg);  // 数组名会隐式转换为指针
  } else if constexpr (std::is_same_v<std::string, T>) {
    printf("%s", arg.c_str());
  } else {
    static_assert(always_false<T>, "Unsupported type");
  }
}

template <typename T, typename... Args>
FUNCTION_ATTRIBUTE void print_args(const T &first, const Args &...rest) {
  print_arg(first);
  if constexpr (sizeof...(rest) > 0) {
    printf(",\t");
    print_args(rest...);
  }
}

// 小端序，按位打印单个参数
template <typename T>
FUNCTION_ATTRIBUTE void print_arg_bit(const T &arg) {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&arg);
  for (int i = sizeof(T) - 1; i >= 0; --i) {
    for (int j = 7; j >= 0; --j) {
      printf("%d", (bytes[i] >> j) & 1);
    }
    if (i > 0) printf(" ");
  }
}

// 多参数按位打印
template <typename T, typename... Args>
FUNCTION_ATTRIBUTE void print_args_bit(const T &first, const Args &...rest) {
  print_arg_bit(first);
  if constexpr (sizeof...(rest) > 0) {
    printf(",\t");
    print_args_bit(rest...);
  }
}

// 按值打印
#define PRINT(...)                 \
  do {                             \
    PRINT_HEADER("", __VA_ARGS__); \
    print_args(__VA_ARGS__);       \
    printf("\n");                  \
  } while (0);

// 按位打印
#define PRINT_BIT(...)                  \
  do {                                  \
    PRINT_HEADER("(bit)", __VA_ARGS__); \
    print_args_bit(__VA_ARGS__);        \
    printf("\n");                       \
  } while (0);
