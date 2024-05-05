use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn implement_trait_for_type(input: TokenStream) -> TokenStream {
    let mut inputs = input.into_iter();
    let trait_name: syn::Ident = inputs.next().unwrap().into();
    let type_name: syn::Ident = inputs.next().unwrap().into();

    let trait_definition = syn::parse_macro_input!(trait_name.clone() as syn::ItemTrait);

    let mut trait_functions = Vec::new();

    if let syn::Item::Trait(trait_item) = trait_definition {
        for item in trait_item.items {
            if let syn::TraitItem::Method(method) = item {
                let func_name = method.sig.ident;
                trait_functions.push(func_name);
            }
        }
    }

    let expanded = quote! {
        impl #trait_name for #type_name {
            #( fn #trait_functions(&self, args: ArgsType) -> ReturnType {
                self.#trait_functions(args)
            } )*
        }
    };

    TokenStream::from(expanded)
}
