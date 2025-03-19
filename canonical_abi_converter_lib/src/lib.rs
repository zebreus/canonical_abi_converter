#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(str_from_utf16_endian)]

// def despecialize(t):
//   match t:
//     case TupleType(ts)       : return RecordType([ FieldType(str(i), t) for i,t in enumerate(ts) ])
//     case EnumType(labels)    : return VariantType([ CaseType(l, None) for l in labels ])
//     case OptionType(t)       : return VariantType([ CaseType("none", None), CaseType("some", t) ])
//     case ResultType(ok, err) : return VariantType([ CaseType("ok", ok), CaseType("error", err) ])
//     case _                   : return t
enum ConverterError {
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

struct SampleContext {
    memory: &'static [u8],
}
enum ContextStringEncoding {
    Utf8,
    Utf16,
    Latin1Utf16,
}
enum RealStringEncoding {
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

trait Context {
    fn string_encoding(&self) -> ContextStringEncoding;
    fn get_array<const SIZE: usize>(&self, offset: u32) -> Result<&[u8; SIZE], ConverterError>;
    fn get_slice(&self, offset: u32, size: usize) -> Result<&[u8], ConverterError>;
}

impl Context for SampleContext {
    fn get_array<const SIZE: usize>(&self, offset: u32) -> Result<&[u8; SIZE], ConverterError> {
        self.memory[offset as usize..offset as usize + SIZE]
            .try_into()
            .map_err(|_| ConverterError::SomethingWentWrong)
    }
    fn get_slice(&self, offset: u32, size: usize) -> Result<&[u8], ConverterError> {
        self.memory[offset as usize..offset as usize + size]
            .try_into()
            .map_err(|_| ConverterError::SomethingWentWrong)
    }
    fn string_encoding(&self) -> ContextStringEncoding {
        ContextStringEncoding::Utf8
    }
}

macro_rules! load_int {
    ($context:expr, $offset:expr, $ty:ty) => {
        Ok(<$ty>::from_le_bytes(
            *($context.get_array::<{ <$ty>::SIZE as usize }>($offset)?),
        ))
    };
}

fn convert_int_to_bool(i: i32) -> Result<bool, ConverterError> {
    if i < 0 {
        return Err(ConverterError::ValueForBoolOutOfRange);
    }
    return Ok(i != 0);
}

fn convert_i32_to_char<C: Context>(_context: &C, codepoint: u32) -> Result<char, ConverterError> {
    if codepoint < 0 || codepoint > 0x10FFFF {
        return Err(ConverterError::CharOutOfRange);
    }
    if codepoint >= 0xD800 && codepoint <= 0xDFFF {
        return Err(ConverterError::CharIsSurrogate);
    }
    Ok(char::from_u32(codepoint).ok_or(ConverterError::CharOutOfRange)?)
}

const CANONICAL_FLOAT32_NAN: u32 = 0x7fc00000;
const CANONICAL_FLOAT64_NAN: u64 = 0x7ff8000000000000;
fn canonicalize_nan32(value: f32) -> f32 {
    if value.is_nan() {
        return f32::from_bits(CANONICAL_FLOAT32_NAN);
    }
    value
}
fn canonicalize_nan64(value: f64) -> f64 {
    if value.is_nan() {
        return f64::from_bits(CANONICAL_FLOAT64_NAN);
    }
    value
}
fn decode_i64_as_float<C: Context>(_context: &C, codepoint: i64) -> Result<f64, ConverterError> {
    Ok(canonicalize_nan64(core_f64_reinterpret_i64(codepoint)))
}
fn decode_i32_as_float<C: Context>(_context: &C, codepoint: i32) -> Result<f32, ConverterError> {
    Ok(canonicalize_nan32(core_f32_reinterpret_i32(codepoint)))
}
fn core_f64_reinterpret_i64(value: i64) -> f64 {
    f64::from_bits(value as u64)
}
fn core_f32_reinterpret_i32(value: i32) -> f32 {
    f32::from_bits(value as u32)
}

fn load_string<C: Context>(_context: &C, offset: u32) -> Result<String, ConverterError> {
    let begin = u32::from_le_bytes(*_context.get_array::<4>(offset as u32)?);
    let tagged_code_units = u32::from_le_bytes(*_context.get_array::<4>(offset + 4)?);
    load_string_from_range(_context, begin, tagged_code_units)
}

const UTF16_TAG: u32 = 1 << 31;
fn load_string_from_range<C: Context>(
    _context: &C,
    offset: u32,
    tagged_code_units: u32,
) -> Result<String, ConverterError> {
    let (alignment, byte_length, encoding) = match _context.string_encoding() {
        ContextStringEncoding::Utf8 => (1, tagged_code_units, RealStringEncoding::Utf8),

        ContextStringEncoding::Utf16 => (2, 2 * tagged_code_units, RealStringEncoding::Utf16Le),
        ContextStringEncoding::Latin1Utf16 => (
            2,
            if tagged_code_units & UTF16_TAG != 0 {
                2 * (tagged_code_units ^ UTF16_TAG)
            } else {
                tagged_code_units
            },
            if tagged_code_units & UTF16_TAG != 0 {
                RealStringEncoding::Utf16Le
            } else {
                RealStringEncoding::Latin1
            },
        ),
    };
    if offset != align_to(offset, alignment) {
        return Err(ConverterError::StringAlignmentError);
    }
    let data = _context.get_slice(offset, byte_length as usize)?;

    encoding.decode(data)
}

trait CanonicalAbi
where
    Self: Sized,
{
    const ALIGNMENT: u32;
    const SIZE: u32;

    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError>;
}
trait CanonicalAbiEnum {}

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
        convert_i32_to_char(context, load_int!(context, offset, u32)?)
    }
}
impl CanonicalAbi for String {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_string(context, offset)
    }
}

const fn alignment_list<T: CanonicalAbi, const MAYBE_LENGTH: isize>() -> u32 {
    // def alignment_list(elem_type, maybe_length):
    if MAYBE_LENGTH > 0 {
        return T::ALIGNMENT;
    }
    return 4;
}
const fn size_list<T: CanonicalAbi, const MAYBE_LENGTH: isize>() -> u32 {
    // def alignment_list(elem_type, maybe_length):
    if MAYBE_LENGTH > 0 {
        return MAYBE_LENGTH as u32 * T::SIZE;
    }
    return 8;
}

const fn alignment_flags<T: Sized>() -> u32 {
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
const fn elem_size_flags<T: Sized>() -> u32 {
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

// Represents discriminant_type in Python
const fn discriminant_type_size<const N: usize>() -> u32 {
    assert!(N > 0 && N < (1 << 32));
    if N <= 256 {
        return 1;
    }
    if N <= 65536 {
        return 2;
    }
    return 4;
}

// def max_case_alignment(cases):
//   a = 1
//   for c in cases:
//     if c.t is not None:
//       a = max(a, alignment(c.t))
//   return a
const fn max_case_alignment<T: CanonicalAbi>() -> u32 {
    return T::ALIGNMENT;
}

fn load_list<'a, C: Context, T: CanonicalAbi, const LENGTH: i32>(
    context: &'a C,
    offset: u32,
) -> Result<Vec<T>, ConverterError> {
    if LENGTH > 0 {
        return load_list_from_valid_range(context, offset, LENGTH as u32);
    }
    let begin = load_int!(context, offset, u32)?;
    let length = load_int!(context, offset + 4, u32)?;
    return load_list_from_range(context, begin, length);
}
fn load_list_from_valid_range<'a, C: Context, T: CanonicalAbi>(
    context: &'a C,
    offset: u32,
    length: u32,
) -> Result<Vec<T>, ConverterError> {
    let mut result = Vec::with_capacity(length as usize);
    let mut current_offset = offset;
    for _ in 0..length {
        result.push(T::load(context, current_offset)?);
        current_offset += T::SIZE;
    }
    Ok(result)
}
fn load_list_from_range<'a, C: Context, T: CanonicalAbi>(
    context: &'a C,
    offset: u32,
    length: u32,
) -> Result<Vec<T>, ConverterError> {
    if offset != align_to(offset, T::ALIGNMENT) {
        return Err(ConverterError::ListAlignmentError);
    }
    load_list_from_valid_range(context, offset, length)
}

impl<T: CanonicalAbi> CanonicalAbi for Vec<T> {
    const ALIGNMENT: u32 = { alignment_list::<T, -1>() };
    const SIZE: u32 = { size_list::<T, -1>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_list::<C, T, -1>(context, offset)
    }
}

const fn align_to(ptr: u32, alignment: u32) -> u32 {
    return (ptr).div_ceil(alignment) * alignment;
}

struct WrappedBitflags<A, T: bitflags::Flags<Bits = A>> {
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

const fn max(current_max: u32, numbers: &[u32]) -> u32 {
    let Some((head, tail)) = numbers.split_first() else {
        return current_max;
    };
    if current_max > *head {
        return max(current_max, tail);
    } else {
        return max(*head, tail);
    }
}

macro_rules! alignment_variant {
    ($length:literal, $($name:tt),*) => {
        {
            let alignments = [
                discriminant_type_size::<$length>(),
                $(max_case_alignment::<$name>(),)*
            ];
            max(0, &alignments)
        }
    };
}
macro_rules! elem_size_variant {
    ($self:ident, $length:literal, $($name:tt),*) => {
            {
                let alignments = [
                    $(max_case_alignment::<$name>(),)*
                ];
                let max_case_alignment = max(0, &alignments);
                let s = discriminant_type_size::<$length>();
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

impl<T: CanonicalAbi> CanonicalAbi for Option<T> {
    const ALIGNMENT: u32 = alignment_variant!(2, (), T);
    const SIZE: u32 = elem_size_variant!(Self, 2, (), T);

    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let discriminant_size = discriminant_type_size::<2>();
        let case_index = match discriminant_size {
            1 => load_int!(context, offset, u8)? as u32,
            2 => load_int!(context, offset, u16)? as u32,
            4 => load_int!(context, offset, u32)?,
            _ => panic!(),
        };
        let offset = offset + discriminant_size;
        if case_index >= 2 {}
        match case_index {
            0 => Ok(Self::None),
            1 => Ok(Self::Some(T::load(context, offset)?)),
            _ => Err(ConverterError::InvalidDiscriminantIndex),
        }
    }
}
impl<T: CanonicalAbi, E: CanonicalAbi> CanonicalAbi for Result<T, E> {
    const ALIGNMENT: u32 = alignment_variant!(2, T, E);
    const SIZE: u32 = elem_size_variant!(Self, 2, (), T);
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let discriminant_size = discriminant_type_size::<2>();
        let case_index = match discriminant_size {
            1 => load_int!(context, offset, u8)? as u32,
            2 => load_int!(context, offset, u16)? as u32,
            4 => load_int!(context, offset, u32)?,
            _ => panic!(),
        };
        let offset = offset + discriminant_size;
        if case_index >= 2 {}
        match case_index {
            0 => Ok(Self::Ok(T::load(context, offset)?)),
            1 => Ok(Self::Err(E::load(context, offset)?)),
            _ => Err(ConverterError::InvalidDiscriminantIndex),
        }
    }
}

macro_rules! max_alignment {
    ($($name:ident),*) => {
        {
                let alignments = [
                    $($name::ALIGNMENT,)*
                ];
                max(0, &alignments)
            }
    };
}
macro_rules! impl_canonical_abi_for_tuple {
    ($($name:ident),*) => {
        impl<$($name: CanonicalAbi),*> CanonicalAbi for ($($name,)*) {
            const ALIGNMENT: u32 = max_alignment!($($name),*);
            const SIZE: u32 = elem_size_record!(Self, $($name),*);
            fn load<'a, CC: Context>(context: &'a CC, offset: u32) -> Result<Self, ConverterError> {



                Ok(( $({
                    let offset = align_to(offset, $name::ALIGNMENT);
                    let v = $name::load(context, offset)?;
                    #[allow(unused)]
                    let offset = offset + $name::SIZE;
                    v
                },)* ))
            }
        }
    };
}

const fn elem_size_record(current_size: u32, fields: &[(u32, u32)], alignment: u32) -> u32 {
    let Some((head, tail)) = fields.split_first() else {
        return align_to(current_size, alignment);
    };
    let (head_size, head_alignment) = head;
    let current_size = align_to(current_size, *head_alignment);
    let current_size = current_size + *head_size;
    return elem_size_record(current_size, tail, alignment);
}

macro_rules! elem_size_record {
    ($self:ident, $($name:ident),*) => {
        {
            let alignments = [
                $(($name::SIZE,$name::ALIGNMENT,),)*
            ];
            elem_size_record(0, &alignments, $self::ALIGNMENT)
        }
    };
}

impl<A: CanonicalAbi> CanonicalAbi for (A,) {
    const ALIGNMENT: u32 = max_alignment!(A);
    const SIZE: u32 = elem_size_record!(Self, A);
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let offset = align_to(offset, A::ALIGNMENT);
        let a = A::load(context, offset)?;
        Ok((a,))
    }
}
impl<A: CanonicalAbi, B: CanonicalAbi> CanonicalAbi for (A, B) {
    const ALIGNMENT: u32 = max_alignment!(A, B);
    const SIZE: u32 = elem_size_record!(Self, A, B);
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let offset = align_to(offset, A::ALIGNMENT);
        let a = A::load(context, offset)?;
        let offset = offset + A::SIZE;

        let offset = align_to(offset, B::ALIGNMENT);
        let b = B::load(context, offset)?;

        Ok((a, b))
    }
}

impl_canonical_abi_for_tuple!(A, B, C);
impl_canonical_abi_for_tuple!(A, B, C, D);
impl_canonical_abi_for_tuple!(A, B, C, D, E);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H);

// impl CanonicalAbi for ErrorContext {
//     const ALIGNMENT: usize = 4;
// }

// fn alignment(t: &Type) -> usize {
//     match t {
//         Type::ErrorContext => 4,
//         Type::Record(fields) => alignment_record(fields),
//         Type::Variant(cases) => alignment_variant(cases),
//         Type::Flags(labels) => alignment_flags(labels),
//         Type::Own | Type::Borrow => 4,
//         Type::Stream | Type::Future => 4,

//     }
// }
