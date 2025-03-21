#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(str_from_utf16_endian)]
#![feature(macro_metavar_expr)]

pub use alignment::{alignment, alignment_flags, alignment_list, discriminant_type};
pub use element_size::{align_to, elem_size, elem_size_flags, elem_size_list};
pub use loading::{
    canonicalize_nan32, canonicalize_nan64, convert_i32_to_char, convert_int_to_bool,
    core_f32_reinterpret_i32, core_f64_reinterpret_i64, decode_i32_as_float, decode_i64_as_float,
    lift_async_value, lift_borrow, lift_error_context, lift_future, lift_own, lift_stream, load,
    load_list, load_list_from_range, load_list_from_valid_range, load_string,
    load_string_from_range,
};
mod alignment;
mod element_size;
mod loading;

pub const fn max(current_max: u32, numbers: &[u32]) -> u32 {
    let Some((head, tail)) = numbers.split_first() else {
        return current_max;
    };
    if current_max > *head {
        return max(current_max, tail);
    } else {
        return max(*head, tail);
    }
}

#[derive(Debug, Clone)]
pub enum ConverterError {
    OutOfBoundsMemoryAccess {
        accessed: u32,
        size: u32,
    },
    SomethingWentWrong,
    CharOutOfRange,
    CharIsSurrogate,
    ValueForBoolOutOfRange,
    /// String pointer is not aligned
    StringAlignmentError,
    /// List pointer is not aligned
    ListAlignmentError,
    // TODO: Support for Latin1 encoded strings
    Latin1EncodedStringsNotSupported,
    // Invalid discriminant index
    InvalidDiscriminantIndex,
}

pub enum ContextStringEncoding {
    Utf8,
    Utf16,
    Latin1Utf16,
}
pub enum RealStringEncoding {
    Utf8,
    Utf16Le,
    Latin1,
}
impl RealStringEncoding {
    fn decode(&self, data: &[u8]) -> Result<String, ConverterError> {
        match self {
            RealStringEncoding::Utf8 => Ok(String::from_utf8_lossy(data).to_string()),
            RealStringEncoding::Utf16Le => Ok(String::from_utf16le_lossy(data)),
            RealStringEncoding::Latin1 => Err(ConverterError::Latin1EncodedStringsNotSupported),
        }
    }
}

pub trait Context {
    fn string_encoding(&self) -> ContextStringEncoding;
    fn get_array<const SIZE: usize>(&self, offset: u32) -> Result<&[u8; SIZE], ConverterError>;
    fn get_slice(&self, offset: u32, size: usize) -> Result<&[u8], ConverterError>;
}

pub trait CanonicalAbi
where
    Self: Sized,
{
    const ALIGNMENT: u32;
    const SIZE: u32;

    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError>;
}

impl CanonicalAbi for () {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 0;
    fn load<'a, C: Context>(_context: &'a C, _offset: u32) -> Result<Self, ConverterError> {
        Ok(())
    }
}
impl CanonicalAbi for bool {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 1;
    // fn load(_data: &[u8; Self::SIZE]) -> Self {
    //     true
    // }
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        convert_int_to_bool(load_int!(context, offset, i32)?)
    }
}
impl CanonicalAbi for u8 {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 1;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i8 {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 1;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for u16 {
    const ALIGNMENT: u32 = 2;
    const SIZE: u32 = 2;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i16 {
    const ALIGNMENT: u32 = 2;
    const SIZE: u32 = 2;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for u32 {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i32 {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for u64 {
    const ALIGNMENT: u32 = 8;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i64 {
    const ALIGNMENT: u32 = 8;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for f32 {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        decode_i32_as_float(context, load_int!(context, offset, i32)?)
    }
}
impl CanonicalAbi for f64 {
    const ALIGNMENT: u32 = 8;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        decode_i64_as_float(context, load_int!(context, offset, i64)?)
    }
}

impl CanonicalAbi for char {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        convert_i32_to_char(context, load_int!(context, offset, i32)?)
    }
}
impl CanonicalAbi for String {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_string(context, offset)
    }
}

impl<T: CanonicalAbi> CanonicalAbi for Vec<T> {
    const ALIGNMENT: u32 = { alignment_list::<T, -1>() };
    const SIZE: u32 = { elem_size_list::<T, -1>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_list::<C, T, -1>(context, offset)
    }
}

pub struct WrappedBitflags<A, T: bitflags::Flags<Bits = A>> {
    pub inner: T,
}
impl<T: bitflags::Flags<Bits = u8>> CanonicalAbi for WrappedBitflags<u8, T> {
    const ALIGNMENT: u32 = { alignment_flags::<T::Bits>() };
    const SIZE: u32 = { elem_size_flags::<T::Bits>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        Ok(WrappedBitflags {
            inner: T::from_bits_truncate(load_int!(context, offset, T::Bits)?),
        })
    }
}
impl<T: bitflags::Flags<Bits = u16>> CanonicalAbi for WrappedBitflags<u16, T> {
    const ALIGNMENT: u32 = { alignment_flags::<T::Bits>() };
    const SIZE: u32 = { elem_size_flags::<T::Bits>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        Ok(WrappedBitflags {
            inner: T::from_bits_truncate(load_int!(context, offset, T::Bits)?),
        })
    }
}
impl<T: bitflags::Flags<Bits = u32>> CanonicalAbi for WrappedBitflags<u32, T> {
    const ALIGNMENT: u32 = { alignment_flags::<T::Bits>() };
    const SIZE: u32 = { elem_size_flags::<T::Bits>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        Ok(WrappedBitflags {
            inner: T::from_bits_truncate(load_int!(context, offset, T::Bits)?),
        })
    }
}

impl<T: CanonicalAbi> CanonicalAbi for Option<T> {
    const ALIGNMENT: u32 = alignment_variant!(2, (), T);
    const SIZE: u32 = elem_size_variant!(Self, 2, (), T);

    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let value = load_variant!(
            context,
            offset,
            enum Self {
                None,
                Some(T),
            }
        );
        Ok(value)
    }
}

impl<T: CanonicalAbi, E: CanonicalAbi> CanonicalAbi for Result<T, E> {
    const ALIGNMENT: u32 = alignment_variant!(2, T, E);
    const SIZE: u32 = elem_size_variant!(Self, 2, (), T);
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let value = load_variant!(
            context,
            offset,
            enum Self {
                Ok(T),
                Err(E),
            }
        );
        Ok(value)
    }
}

macro_rules! impl_canonical_abi_for_tuple {
    ($($name:ident),*) => {
        impl<$($name: CanonicalAbi),*> CanonicalAbi for ($($name,)*) {
            const ALIGNMENT: u32 = alignment_record!($($name),*);
            const SIZE: u32 = elem_size_record!(Self, $($name),*);
            fn load<'a, CC: Context>(context: &'a CC, offset: u32) -> Result<Self, ConverterError> {
                Ok(load_tuple!(context, offset, ($($name,)*)))
            }
        }
    };
}

impl_canonical_abi_for_tuple!(A);
impl_canonical_abi_for_tuple!(A, B);
impl_canonical_abi_for_tuple!(A, B, C);
impl_canonical_abi_for_tuple!(A, B, C, D);
impl_canonical_abi_for_tuple!(A, B, C, D, E);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[derive(Debug, Clone)]
    pub struct SampleContext {
        memory: Vec<u8>,
    }

    impl SampleContext {
        pub fn new(memory: Vec<u8>) -> Self {
            SampleContext { memory }
        }
    }

    impl Context for SampleContext {
        fn get_array<const SIZE: usize>(&self, offset: u32) -> Result<&[u8; SIZE], ConverterError> {
            if offset as usize + SIZE > self.memory.len() {
                return Err(ConverterError::OutOfBoundsMemoryAccess {
                    accessed: (offset as usize + SIZE) as u32,
                    size: self.memory.len() as u32,
                });
            }
            self.memory[offset as usize..offset as usize + SIZE]
                .try_into()
                .map_err(|_| ConverterError::SomethingWentWrong)
        }
        fn get_slice(&self, offset: u32, size: usize) -> Result<&[u8], ConverterError> {
            if offset as usize + size > self.memory.len() {
                return Err(ConverterError::OutOfBoundsMemoryAccess {
                    accessed: (offset as usize + size) as u32,
                    size: self.memory.len() as u32,
                });
            }
            self.memory[offset as usize..offset as usize + size]
                .try_into()
                .map_err(|_| ConverterError::SomethingWentWrong)
        }
        fn string_encoding(&self) -> ContextStringEncoding {
            ContextStringEncoding::Utf8
        }
    }
}
