//! This module contains the implementation of the alignment of the canonical ABI.
//!
//! Some of the alignment calculations need to be defined with the trait implementations, these are not here.
//!
//! The structure of the module is closely modeled after the reference python implementation from the [canonical ABI explainer](https://github.com/WebAssembly/component-model/blob/6e08e28368fe6301be956ece68d08077ebc09423/design/mvp/CanonicalABI.md#alignment). We aim to have a equivalent implementation of most of the functions in the python implementation. Some will need to be implemented as macros, but the goal is to stay close to the python implementation. Every python function that does not have a direct equivalent, will be mentioned in a comment.

use crate::CanonicalAbi;

/// Each value type is assigned an alignment which is used by subsequent Canonical ABI definitions. Presenting the definition of alignment piecewise, we start with the top-level case analysis:
///
/// ```python
/// def alignment(t):
///   match despecialize(t):
///     case BoolType()                  : return 1
///     case S8Type() | U8Type()         : return 1
///     case S16Type() | U16Type()       : return 2
///     case S32Type() | U32Type()       : return 4
///     case S64Type() | U64Type()       : return 8
///     case F32Type()                   : return 4
///     case F64Type()                   : return 8
///     case CharType()                  : return 4
///     case StringType()                : return 4
///     case ErrorContextType()          : return 4
///     case ListType(t, l)              : return alignment_list(t, l)
///     case RecordType(fields)          : return alignment_record(fields)
///     case VariantType(cases)          : return alignment_variant(cases)
///     case FlagsType(labels)           : return alignment_flags(labels)
///     case OwnType() | BorrowType()    : return 4
///     case StreamType() | FutureType() : return 4
/// ```
///
/// Is represented by the `CanonicalAbi` trait, which has a `const ALIGNMENT: u32` associated constant. Because of that this function is basically a no-op.
pub const fn alignment<T: CanonicalAbi>() -> u32 {
    return T::ALIGNMENT;
}

/// List alignment is the same as tuple alignment when the length is fixed and otherwise uses the alignment of pointers.
///
/// ```python
/// def alignment_list(elem_type, maybe_length):
///   if maybe_length is not None:
///     return alignment(elem_type)
///   return 4
/// ```
pub const fn alignment_list<T: CanonicalAbi, const MAYBE_LENGTH: isize>() -> u32 {
    // def alignment_list(elem_type, maybe_length):
    if MAYBE_LENGTH > 0 {
        return T::ALIGNMENT;
    }
    return 4;
}

/// Record alignment is tuple alignment
///
/// ```python
/// def alignment_record(fields):
///   a = 1
///   for f in fields:
///     a = max(a, alignment(f.t))
///   return a
/// ```
#[macro_export]
macro_rules! alignment_record {
    ($($name:ident),*) => {{
        let alignments = [
            $($name::ALIGNMENT,)*
        ];
        max(1, &alignments)
    }};
}

/// ```python
/// def alignment_variant(cases):
///   return max(alignment(discriminant_type(cases)), max_case_alignment(cases))
/// ```
///
/// We dont need to get the alignment of the discriminant here, because our `discriminant_type` function already returns the size/alignment of the discriminant.
#[macro_export]
macro_rules! alignment_variant {
    ($length:literal, $($name:tt),*) => {
        {
            let alignments = [
                discriminant_type::<$length>(),
                max_case_alignment!($($name),*)
            ];
            max(0, &alignments)
        }
    };
}

/// ```python
/// def discriminant_type(cases):
///   n = len(cases)
///   assert(0 < n < (1 << 32))
///   match math.ceil(math.log2(n)/8):
///     case 0: return U8Type()
///     case 1: return U8Type()
///     case 2: return U16Type()
///     case 3: return U32Type()
/// ```
///
/// This is a bit different from the python implementation in that it returns a size instead of a type. We cant really return types in Rust and I didnt want to use an enum here.
pub const fn discriminant_type<const N: usize>() -> u32 {
    assert!(N > 0 && N < (1 << 32));
    if N <= 256 {
        return 1;
    }
    if N <= 65536 {
        return 2;
    }
    return 4;
}

/// ```python
/// def max_case_alignment(cases):
///   a = 1
///   for c in cases:
///     if c.t is not None:
///       a = max(a, alignment(c.t))
///   return a
/// ```
#[macro_export]
macro_rules! max_case_alignment {
    ($($name:tt),*) => {
        {
            let alignments = [
                $(<$name>::ALIGNMENT,)*
            ];
            max(1, &alignments)
        }
    };
}

/// ```python
/// def alignment_flags(labels):
///   n = len(labels)
///   assert(0 < n <= 32)
///   if n <= 8: return 1
///   if n <= 16: return 2
///   return 4
/// ```
///
pub const fn alignment_flags<T: Sized>() -> u32 {
    let max_labels = size_of::<T>() * 8;
    assert!(max_labels > 0 && max_labels <= 32);
    if max_labels <= 8 {
        return 1;
    }
    if max_labels <= 16 {
        return 2;
    }
    return 4;
}
