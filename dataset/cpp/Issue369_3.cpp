struct foo{
  constexpr foo() noexcept {};
};


const foo& create(){
	static foo value = foo();
  	return value;
}
