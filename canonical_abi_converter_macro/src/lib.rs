#[proc_macro_attribute]
pub fn main(
    _args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {

    return input.into();
}