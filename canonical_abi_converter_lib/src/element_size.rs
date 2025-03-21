//! This module contains the implementation of the element size section of the canonical ABI.
//!
//! Each value type is also assigned an elem_size which is the number of bytes used when values of the type are stored as elements of a list. Having this byte size be a static property of the type instead of attempting to use a variable-length element-encoding scheme both simplifies the implementation and maps well to languages which represent lists as random-access arrays. Empty types, such as records with no fields, are not permitted, to avoid complications in source languages.
//!
//! The structure of the module is closely modeled after the reference python implementation from the [canonical ABI explainer](https://github.com/WebAssembly/component-model/blob/6e08e28368fe6301be956ece68d08077ebc09423/design/mvp/CanonicalABI.md#element-size). We aim to have a equivalent implementation of most of the functions in the python implementation. Some will need to be implemented as macros, but the goal is to stay close to the python implementation. Every python function that does not have a direct equivalent, will be mentioned in a comment.

use crate::CanonicalAbi;

/// Each value type is also assigned an elem_size which is the number of bytes used when values of the type are stored as elements of a list. Having this byte size be a static property of the type instead of attempting to use a variable-length element-encoding scheme both simplifies the implementation and maps well to languages which represent lists as random-access arrays. Empty types, such as records with no fields, are not permitted, to avoid complications in source languages.
///
/// ```python
/// def elem_size(t):
///   match despecialize(t):
///     case BoolType()                  : return 1
///     case S8Type() | U8Type()         : return 1
///     case S16Type() | U16Type()       : return 2
///     case S32Type() | U32Type()       : return 4
///     case S64Type() | U64Type()       : return 8
///     case F32Type()                   : return 4
///     case F64Type()                   : return 8
///     case CharType()                  : return 4
///     case StringType()                : return 8
///     case ErrorContextType()          : return 4
///     case ListType(t, l)              : return elem_size_list(t, l)
///     case RecordType(fields)          : return elem_size_record(fields)
///     case VariantType(cases)          : return elem_size_variant(cases)
///     case FlagsType(labels)           : return elem_size_flags(labels)
///     case OwnType() | BorrowType()    : return 4
///     case StreamType() | FutureType() : return 4
/// ```
///
/// The element size represented by the associated constant `SIZE` of `CanonicalAbi`. Because of that this function is basically a no-op.
pub const fn elem_size<T: CanonicalAbi>() -> u32 {
    return T::SIZE;
}

/// ```python
/// def elem_size_list(elem_type, maybe_length):
///   if maybe_length is not None:
///     return maybe_length * elem_size(elem_type)
///   return 8
/// ```
pub const fn elem_size_list<T: CanonicalAbi, const MAYBE_LENGTH: isize>() -> u32 {
    // def alignment_list(elem_type, maybe_length):
    if MAYBE_LENGTH > 0 {
        return MAYBE_LENGTH as u32 * T::SIZE;
    }
    return 8;
}

/// ```python
/// def elem_size_record(fields):
///   s = 0
///   for f in fields:
///     s = align_to(s, alignment(f.t))
///     s += elem_size(f.t)
///   assert(s > 0)
///   return align_to(s, alignment_record(fields))
/// ```
#[macro_export]
macro_rules! elem_size_record {
    ($self:ident, $($name:ident),*) => {
        {
            const fn elem_size_record(current_size: u32, fields: &[(u32, u32)], alignment: u32) -> u32 {
                let Some((head, tail)) = fields.split_first() else {
                    return align_to(current_size, alignment);
                };
                let (head_size, head_alignment) = head;
                let current_size = align_to(current_size, *head_alignment);
                let current_size = current_size + *head_size;
                return elem_size_record(current_size, tail, alignment);
            }

            let alignments = [
                $(($name::SIZE,$name::ALIGNMENT,),)*
            ];
            elem_size_record(0, &alignments, $self::ALIGNMENT)
        }
    };
}

/// ```python
/// def align_to(ptr, alignment):
///   return math.ceil(ptr / alignment) * alignment
/// ```
pub const fn align_to(ptr: u32, alignment: u32) -> u32 {
    return (ptr).div_ceil(alignment) * alignment;
}

/// ```python
/// def elem_size_variant(cases):
///   s = elem_size(discriminant_type(cases))
///   s = align_to(s, max_case_alignment(cases))
///   cs = 0
///   for c in cases:
///     if c.t is not None:
///       cs = max(cs, elem_size(c.t))
///   s += cs
///   return align_to(s, alignment_variant(cases))
/// ```
#[macro_export]
macro_rules! elem_size_variant {
    ($self:ident, $length:literal, $($name:tt),*) => {
            {
                let max_case_alignment = max_case_alignment!($($name),*);
                let s = discriminant_type::<$length>();
                let s = align_to(s, max_case_alignment);
                let element_sizes = [
                    $(<$name>::SIZE,)*
                ];
                let max_element_size = max(0, &element_sizes);
                let s = s + max_element_size;
                let s = align_to(s, $self::ALIGNMENT);
                s
            }
    };
}

/// ```python
/// def elem_size_flags(labels):
///   n = len(labels)
///   assert(0 < n <= 32)
///   if n <= 8: return 1
///   if n <= 16: return 2
///   return 4
/// ```
pub const fn elem_size_flags<T: Sized>() -> u32 {
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
